def kmeans(data, centroids):
    """
    Function implementing the k-means clustering.
    
    :param data
        data
    :param centroids
        initial centroids
    :return
        final centroids
    """
    ### START CODE HERE ### 
    # Initialize container with cluster info
    labels = {k:[] for k in range(len(centroids))}

    for i, point in enumerate(data):
        closest = np.inf
        belongs_to = 0 # index to centroid
        
        # Find closest centroid
        for j, c in enumerate(centroids):
            # Get L2 norm between points
            dist = np.linalg.norm(point - c)
            if dist < closest:
                closest = dist
                belongs_to = j
        labels[belongs_to].append(i)
    
    # Update positions of centroids
    for centroid_index, pt_indices in labels.items():
        centroids[centroid_index] = np.mean(data[pt_indices], axis=0)
    
    ### END CODE HERE ### 
    return centroids

def silhouette_score(data, centroids):
    """
    Function implementing the k-means clustering.
    
    :param data
        data
    :param centroids
        centroids
    :return
        mean Silhouette Coefficient of all samples
    """
    ### START CODE HERE ### 
    s = []
    labels = {k:[] for k in range(len(centroids))}
    for point in data: # Place points to correct centroid
        distances = np.linalg.norm(point - centroids, axis=1)
        labels[np.argmin(distances)].append(point)
    # Iterate through centroids
    for i in range(centroids.shape[0]):
        pts = labels[i]
        for j, pt in enumerate(pts):
            a_dists = np.linalg.norm(pt - pts, axis=1)
            a = np.mean(np.delete(a_dists, j)) # exclude distance to itself
            b_dists = np.linalg.norm(pt - centroids, axis=1)
            b = np.min(np.delete(b_dists, i))
            s.append( (b-a)/max(a,b) )

    score = np.mean(s)
        

    ### END CODE HERE ### 
    return score