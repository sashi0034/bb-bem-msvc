#include "stdafx.h"
#include "Clustering.h"

namespace
{
	double EuclideanDistance(const Vec3& a, const Vec3& b) {
		Vec3 d = a - b;
		return d.length(); // または sqrt(lengthSq()) 相当
	}

	double AverageLinkageDistance(const Array<Vec3>& clusterA, const Array<Vec3>& clusterB) {
		double sum = 0.0;
		int count = 0;
		for (const auto& a : clusterA) {
			for (const auto& b : clusterB) {
				sum += EuclideanDistance(a, b);
				++count;
			}
		}
		return sum / count;
	}

	double SingleLinkageDistance(const Array<Vec3>& clusterA, const Array<Vec3>& clusterB) {
		double minDist = std::numeric_limits<double>::max();
		for (const auto& a : clusterA) {
			for (const auto& b : clusterB) {
				double dist = EuclideanDistance(a, b);
				if (dist < minDist) minDist = dist;
			}
		}
		return minDist;
	}
}

namespace Clustering
{
	Array<int> ClusteringSingleLinkage(const Array<Vec3>& positions, int clusterCount) {
		int n = positions.size();
		Array<Array<int>> clusters(n);
		for (int i = 0; i < n; ++i)
			clusters[i] = {i};

		while (clusters.size() > clusterCount) {
			double minDist = std::numeric_limits<double>::max();
			int mergeA = -1, mergeB = -1;

			for (int i = 0; i < clusters.size(); ++i) {
				for (int j = i + 1; j < clusters.size(); ++j) {
					Array<Vec3> ptsA, ptsB;
					for (int idx : clusters[i]) ptsA.push_back(positions[idx]);
					for (int idx : clusters[j]) ptsB.push_back(positions[idx]);

					double dist = SingleLinkageDistance(ptsA, ptsB);
					if (dist < minDist) {
						minDist = dist;
						mergeA = i;
						mergeB = j;
					}
				}
			}

			// マージ
			clusters[mergeA].insert(clusters[mergeA].end(), clusters[mergeB].begin(), clusters[mergeB].end());
			clusters.erase(clusters.begin() + mergeB);
		}

		Array<int> result(n, -1);
		for (int i = 0; i < clusters.size(); ++i) {
			for (int idx : clusters[i]) result[idx] = i;
		}
		return result;
	}

	Array<int> ClusteringAverageLinkage(const Array<Vec3>& positions, int clusterCount) {
		int n = positions.size();
		Array<Array<int>> clusters(n);
		for (int i = 0; i < n; ++i)
			clusters[i] = {i};

		while (clusters.size() > clusterCount) {
			double minDist = std::numeric_limits<double>::max();
			int mergeA = -1, mergeB = -1;

			for (int i = 0; i < clusters.size(); ++i) {
				for (int j = i + 1; j < clusters.size(); ++j) {
					Array<Vec3> ptsA, ptsB;
					for (int idx : clusters[i]) ptsA.push_back(positions[idx]);
					for (int idx : clusters[j]) ptsB.push_back(positions[idx]);

					double dist = AverageLinkageDistance(ptsA, ptsB);
					if (dist < minDist) {
						minDist = dist;
						mergeA = i;
						mergeB = j;
					}
				}
			}

			// マージ
			clusters[mergeA].insert(clusters[mergeA].end(), clusters[mergeB].begin(), clusters[mergeB].end());
			clusters.erase(clusters.begin() + mergeB);
		}

		Array<int> result(n, -1);
		for (int i = 0; i < clusters.size(); ++i) {
			for (int idx : clusters[i]) result[idx] = i;
		}
		return result;
	}

	Array<int> ClusteringKMeans(const Array<Vec3>& positions, int clusterCount) {
		int n = positions.size();
		Array<Vec3> centroids;
		Array<int> labels(n, 0);

		// 初期セントロイドをランダムに選ぶ
		std::sample(positions.begin(), positions.end(), std::back_inserter(centroids),
		            clusterCount, std::mt19937{std::random_device{}()});

		bool changed = true;
		while (changed) {
			changed = false;

			// 各点を最も近いクラスタに割り当て
			for (int i = 0; i < n; ++i) {
				double minDist = std::numeric_limits<double>::max();
				int best = 0;
				for (int k = 0; k < clusterCount; ++k) {
					double dist = EuclideanDistance(positions[i], centroids[k]);
					if (dist < minDist) {
						minDist = dist;
						best = k;
					}
				}
				if (labels[i] != best) {
					labels[i] = best;
					changed = true;
				}
			}

			// 新しいセントロイドを計算
			Array<Vec3> newCentroids(clusterCount, Vec3{});
			Array<int> counts(clusterCount, 0);

			for (int i = 0; i < n; ++i) {
				newCentroids[labels[i]] += positions[i];
				counts[labels[i]]++;
			}

			for (int k = 0; k < clusterCount; ++k) {
				if (counts[k] > 0) {
					newCentroids[k] /= counts[k];
				}
				else {
					newCentroids[k] = positions[rand() % n]; // 空クラスタ回避
				}
			}

			centroids = newCentroids;
		}

		return labels;
	}
}
