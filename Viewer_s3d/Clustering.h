#pragma once

namespace Clustering
{
	Array<int> ClusteringSingleLinkage(const Array<Vec3>& positions, int clusterCount);

	Array<int> ClusteringAverageLinkage(const Array<Vec3>& positions, int clusterCount);

	Array<int> ClusteringKMeans(const Array<Vec3>& positions, int clusterCount);
}
