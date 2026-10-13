import numpy as np
import networkx as nx
from sklearn.cluster import KMeans


def compute_localization_operator(laplacian, nu=10.0):
    """Compute the sqrt(N)-scaled graph heat-kernel localization operator."""
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

    max_eigenvalue = eigenvalues.max()
    if max_eigenvalue > 0:
        eigenvalues = eigenvalues / max_eigenvalue

    kernel = np.exp(-nu * eigenvalues)
    localization = np.sqrt(laplacian.shape[0]) * (eigenvectors @ np.diag(kernel) @ eigenvectors.T)

    return np.abs(localization)


def extract_sensor_features(data, fft_bins=5):
    """Extract normalized temporal and spectral features for each sensor.

    Parameters
    ----------
    data : ndarray, shape (T, N)
        T time samples from N sensors.
    fft_bins : int
        Number of low-frequency FFT magnitudes to include.

    Returns
    -------
    features : ndarray, shape (N, 9 + fft_bins)
        Min-max normalized feature matrix.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must have shape (T, N).")

    T, N = data.shape
    if T < 3:
        raise ValueError("At least 3 time samples are required for FFT features.")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values.")
    if not isinstance(fft_bins, (int, np.integer)) or fft_bins < 1:
        raise ValueError("fft_bins must be a positive integer.")

    features = np.zeros((N, 9 + fft_bins), dtype=float)

    for sensor_idx in range(N):
        x = data[:, sensor_idx]

        # Statistical features
        mean = np.mean(x)
        maximum = np.max(x)
        minimum = np.min(x)
        variance = np.var(x)
        std = np.std(x)
        skewness = np.mean((x - mean) ** 3) / std**3 if std > 0 else 0.0
        kurtosis = np.mean((x - mean) ** 4) / std**4 - 3.0 if std > 0 else -3.0
        rms = np.sqrt(np.mean(x**2))

        # Spectral features
        # Positive frequencies only, excluding the DC component.
        spectrum = np.abs(np.fft.rfft(x))[1:]
        frequencies = np.fft.rfftfreq(T)[1:]

        peak_idx = np.argmax(spectrum)
        peak_magnitude = spectrum[peak_idx]
        peak_frequency = frequencies[peak_idx]

        low_frequency_magnitudes = np.zeros(fft_bins)
        n_available = min(fft_bins, len(spectrum))
        low_frequency_magnitudes[:n_available] = spectrum[:n_available]

        features[sensor_idx] = np.concatenate([
            [
                mean,
                maximum,
                minimum,
                variance,
                skewness,
                kurtosis,
                rms,
                peak_magnitude,
                peak_frequency,
            ],
            low_frequency_magnitudes,
        ])

    feature_min = features.min(axis=0)
    feature_range = features.max(axis=0) - feature_min
    feature_range[feature_range == 0] = 1.0

    return (features - feature_min) / feature_range


def compute_topology_features(
    graph, nu=10.0, normalization_topofeature=True, betweenness_weight="weight"
):
    """Compute five graph-topology features, optionally standardized.

    betweenness_weight="weight" reproduces the supplied reference code.
    Set it to "distance" to use inverse edge strengths as path lengths.

    Features are:
      1. weighted degree
      2. betweenness centrality
      3. eigenvector centrality
      4. harmonic centrality
      5. localization-operator norm
    """
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise TypeError("graph must be an undirected networkx.Graph.")
    if betweenness_weight not in {"weight", "distance"}:
        raise ValueError("betweenness_weight must be either 'weight' or 'distance'.")
    if not np.isfinite(nu) or nu < 0:
        raise ValueError("nu must be nonnegative and finite.")

    nodes = list(graph.nodes())

    # Work with explicit weights; unweighted edges are treated as weight 1.
    weighted_graph = graph.copy()
    for _, _, edge_data in weighted_graph.edges(data=True):
        edge_data.setdefault("weight", 1.0)
        weight = edge_data["weight"]
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("All graph edge weights must be finite and positive.")

    distance_graph = weighted_graph.copy()
    for _, _, edge_data in distance_graph.edges(data=True):
        weight = edge_data["weight"]
        edge_data["distance"] = 1.0 / weight

    laplacian = nx.laplacian_matrix(
        weighted_graph,
        nodelist=nodes,
        weight="weight",
    ).toarray()

    weighted_degree = np.array([
        sum(edge_data["weight"] for _, _, edge_data in weighted_graph.edges(node, data=True))
        for node in nodes
    ])

    betweenness_dict = nx.betweenness_centrality(
        distance_graph,
        weight=betweenness_weight,
    )
    betweenness = np.array([betweenness_dict[node] for node in nodes])

    eigenvector_dict = nx.eigenvector_centrality(
        weighted_graph,
        weight="weight",
        tol=1e-3,
        max_iter=1000,
    )
    eigenvector = np.array([eigenvector_dict[node] for node in nodes])

    harmonic_dict = nx.harmonic_centrality(
        distance_graph,
        distance="distance",
    )
    harmonic = np.array([harmonic_dict[node] for node in nodes])

    localization = compute_localization_operator(laplacian, nu=nu)
    localization_score = np.linalg.norm(localization, axis=1)

    topology = np.column_stack([
        weighted_degree,
        betweenness,
        eigenvector,
        harmonic,
        localization_score,
    ])

    if normalization_topofeature:
        mean = topology.mean(axis=0)
        std = topology.std(axis=0)
        std[std == 0] = 1.0
        topology = (topology - mean) / std

    return topology


def TVML(
    data,
    graph,
    num_sensors,
    alpha=(0.2, 0.2, 0.2, 0.2, 0.2),
    random_state=42,
    nu=10.0,
    normalization_topofeature=True,
    return_indices=False,
    betweenness_weight="weight",
):
    """Select representative sensors using the TVML method.

    TVML first clusters sensors using temporal/spectral features, then selects
    one sensor from each cluster using a weighted graph-topology score.

    Parameters
    ----------
    data : ndarray, shape (T, N)
        Sensor time series. Columns must follow the same order as graph.nodes().
    graph : networkx.Graph
        Sensor graph.
    num_sensors : int
        Number of sensors to select.
    alpha : sequence of 5 floats
        Weights for the five topology features.
    random_state : int
        KMeans random seed.
    nu : float
        Localization-kernel decay parameter.
    normalization_topofeature : bool
        If True, z-score normalize topology features column-wise.
        If False, use the raw topology features.
    return_indices : bool
        If True, also return zero-based selected column indices.
    betweenness_weight : {"weight", "distance"}
        Path-length attribute for betweenness centrality. "weight" matches
        the supplied reference implementation. "distance" treats edge
        weights as connection strengths and uses their reciprocals.

    Returns
    -------
    selected_labels : ndarray
        Selected graph node labels.
    selected_indices : ndarray, optional
        Returned only when return_indices=True.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must have shape (T, N).")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values.")

    _, num_nodes = data.shape
    node_list = list(graph.nodes())
    node_labels = np.asarray(node_list)
    if node_labels.ndim != 1:
        node_labels = np.empty(len(node_list), dtype=object)
        node_labels[:] = node_list

    if len(node_labels) != num_nodes:
        raise ValueError("The number of graph nodes must equal the number of data columns.")
    if not 1 <= num_sensors <= num_nodes:
        raise ValueError("num_sensors must be between 1 and the number of sensors.")

    alpha = np.asarray(alpha, dtype=float)
    if alpha.shape != (5,):
        raise ValueError("alpha must contain 5 weights.")
    if not np.all(np.isfinite(alpha)):
        raise ValueError("alpha weights must be finite.")

    # 1. Temporal/spectral sensor features and clustering
    sensor_features = extract_sensor_features(data)
    if np.unique(sensor_features, axis=0).shape[0] < num_sensors:
        raise ValueError(
            "Cannot form the requested number of distinct clusters from sensor features."
        )
    clusters = KMeans(
        n_clusters=num_sensors,
        random_state=random_state,
        n_init="auto",
    ).fit_predict(sensor_features)

    # 2. Relabel graph so graph-node order matches data-column order
    node_to_index = {label: idx for idx, label in enumerate(node_labels)}
    mapped_graph = nx.relabel_nodes(graph, node_to_index, copy=True)

    # 3. Graph topology score
    topology_features = compute_topology_features(
        mapped_graph,
        nu=nu,
        normalization_topofeature=normalization_topofeature,
        betweenness_weight=betweenness_weight,
    )
    topology_scores = topology_features @ alpha

    # 4. Pick highest-topology-score sensor in each temporal cluster
    selected_indices = []

    for cluster_id in range(num_sensors):
        members = np.flatnonzero(clusters == cluster_id)
        representative = members[np.argmax(topology_scores[members])]
        selected_indices.append(representative)

    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_labels = node_labels[selected_indices]

    if return_indices:
        return selected_labels, selected_indices

    return selected_labels
