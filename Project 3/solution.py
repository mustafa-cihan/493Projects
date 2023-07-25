import numpy as np

# Read graph data from file
def read_graph(file):
    # Open file for reading
    with open(file, 'r') as f:
        # Read all lines from the file
        lines = f.readlines()

    # Initialize dictionaries for vertices and edges
    vertices = {}
    edges = {}

    # Flag to denote if we are reading vertices or edges
    is_vertex = True

    # Loop over all lines
    for line in lines:
        # Ignore '*Vertices' line
        if line.startswith('*Vertices'):
            continue
        # If '*Edges' line is encountered, set is_vertex to False
        if line.startswith('*Edges'):
            is_vertex = False
            continue
        # If is_vertex is True, read vertex data
        if is_vertex:
            # Parse index and name from line
            index, name = line.strip().split(' ', 1)
            # Add vertex to vertices dictionary
            vertices[int(index)-1] = name.strip('"')
            # Initialize empty list for edges of this vertex
            edges[int(index)-1] = []
        else:  # if is_vertex is False, read edge data
            # Parse vertices connected by this edge
            vertex1, vertex2 = map(int, line.strip().split())
            # Add edge to both vertices, if not already present
            if not (vertex2-1 in edges[vertex1-1]):        
                edges[vertex1-1].append(vertex2-1)
            if not (vertex1-1 in edges[vertex2-1]):
                edges[vertex2-1].append(vertex1-1)

    return vertices, edges

# Compute probability matrix
def compute_prob_matrix(edges, vertices, teleportation):
    # Get number of vertices
    n = len(vertices)
    
    # Initialize probability matrix with teleportation probabilities
    prob_matrix = (teleportation / n) * np.ones((n, n))

    # For each vertex, compute transition probabilities
    for vertex in vertices:
        if len(edges[vertex]) != 0:
            for edge in edges[vertex]:
                # Add transition probability to corresponding cell in matrix
                prob_matrix[vertex][edge] += (1 - teleportation) / len(edges[vertex])

    return prob_matrix

# Implement PageRank algorithm
def page_rank(vertices, prob_matrix, tolerance):
    # Get number of vertices
    n = len(vertices)

    # Initialize ranks with equal probabilities
    ranks = np.full(n, 1/n) 

    # Initialize previous ranks with zeros
    prev_ranks = np.zeros(n)

    # Run PageRank until ranks convergence
    # for this example, we limit it to 10 iterations for simplicity
    for i in range(10):
        # Save current ranks as previous ranks
        prev_ranks = ranks
        # Compute new ranks by multiplying current ranks with probability matrix
        ranks = ranks @ prob_matrix

    # Return ranks as a dictionary, mapping vertex index to its rank
    return {vertex: ranks[vertex] for vertex in vertices}

# Read vertices and edges from file
vertices, edges = read_graph('data.txt')

# Compute probability matrix
prob_matrix = compute_prob_matrix(edges, vertices, 0.10)

# Run PageRank algorithm
ranks = page_rank(vertices, prob_matrix, 1e-6)

# Sort vertices by their ranks, in descending order
sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

# Print the top 20 most important vertices
for i in range(20):
    print(f"{vertices[sorted_ranks[i][0]]}: {sorted_ranks[i][1]}")
