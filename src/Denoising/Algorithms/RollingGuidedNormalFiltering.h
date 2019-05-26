#ifndef ROLLINGGUIDEDNORMALFILTERING_H
#define ROLLINGGUIDEDNORMALFILTERING_H

/*
 @brief: "Rolling Guided Mesh Normal Filtering" class
 @reference: Rolling Guided Mesh Normal Filtering, PG2015
*/

#include "MeshDenoisingBase.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <queue>
#include <utility>
#include <cstdio>
#include <qmatrix.h>
//include lib cuda
#include <ANN/ANN.h>

class RollingGuidedNormalFiltering : public MeshDenoisingBase
{
public:
    RollingGuidedNormalFiltering(DataManager *_data_manager, ParameterSet *_parameter_set);
    ~RollingGuidedNormalFiltering();
private:
    void denoise();
    void initParameters();
    void getVertexBasedFaceNeighbor(TriMesh &mesh, TriMesh::FaceHandle fh, std::vector<TriMesh::FaceHandle> &face_neighbor);
    void getRadiusBasedFaceNeighbor(TriMesh &mesh, TriMesh::FaceHandle fh, double radius, std::vector<TriMesh::FaceHandle> &face_neighbor);
    void getAllFaceNeighborGMNF(TriMesh &mesh, FaceNeighborType face_neighbor_type, double radius, bool include_central_face, std::vector< std::vector<TriMesh::FaceHandle> > &all_face_neighbor);
    void getGlobalMeanEdgeLength(TriMesh &mesh, double& mean_edge_length);
    void getLocalMeanEdgeLength(TriMesh &mesh, std::vector<TriMesh::FaceHandle> &face_neighbor, double &local_mean_edge);
    double GaussianWeight(double distance, double sigma);
	void updateVertexPosition(TriMesh & mesh, std::vector<TriMesh::Normal>& filtered_normals, int iteration_number, bool fixed_boundary, unsigned int ring);
    void getMultipleRingNeighbor(TriMesh &mesh, TriMesh::FaceHandle fh, int n, std::vector<TriMesh::FaceHandle> &face_neighbor);


	void updateVertexPositionWithWeight(TriMesh & mesh, std::vector<TriMesh::Normal>& filtered_normals, int iteration_number, bool fixed_boundary);


    void updateFilteredNormals(TriMesh &mesh, std::vector<TriMesh::Normal> &filtered_normals);
    void updateFilteredNormalsLocalScheme(TriMesh &mesh, std::vector<TriMesh::Normal> &filtered_normals);
    void updateFilteredNormalsGlobalScheme(TriMesh &mesh, std::vector<TriMesh::Normal> &filtered_normals);

	// GPU speed up
	//void gpuUpdateFilteredNormals(TriMesh &mesh, std::vector<TriMesh::Normal> &filtered_normals);
	//void gpuUpdateVertex(TriMesh &mesh, std::vector<TriMesh::Normal> &filtered_normals);

	//get laplacian rtv
	void getGradientNormal(TriMesh &mesh, const std::vector<TriMesh::Normal> &all_face_normal, std::vector<TriMesh::Normal> &gradient_normal)
	{
		gradient_normal.resize(mesh.n_faces());

		for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
		{
			std::vector<TriMesh::FaceHandle> face_neighbor;
			MeshDenoisingBase::getFaceNeighbor(mesh, f_it, kVertexBased, face_neighbor);
			int cnt = 0;
			TriMesh::Normal temp(0.0, 0.0, 0.0);
			for (std::vector<TriMesh::FaceHandle>::iterator it = face_neighbor.begin(); it != face_neighbor.end(); ++it)
			{
				temp += all_face_normal.at(it->idx());
				cnt++;
			}
			gradient_normal.at(f_it->idx()) = temp / cnt - all_face_normal.at(f_it->idx());
		}
	}

	// covariance 
	double getCovariance(const std::vector<int>& idxs2, const std::vector<double>& dists, const std::vector<double>& all_face_area, const std::vector<TriMesh::Normal> &normals, double gaussian)
	{
		TriMesh::Normal mean_normal = TriMesh::Normal(0.0, 0.0, 0.0);
		double normalizer = 0.0;
		for (int i = 0; i < int(idxs2.size()); i++)
		{
			double curr_area = all_face_area[idxs2[i]];
			double spatial_distance = dists[i];
			double g = GaussianWeight(spatial_distance, gaussian);
			TriMesh::Normal nr_jk = normals[idxs2[i]];
			mean_normal += g * curr_area * nr_jk;
			normalizer += g * curr_area;
		}
		mean_normal /= normalizer;

		double var = 0.0;
		for (int i = 0; i < int(idxs2.size()); i++)
		{
			TriMesh::Normal nr_jk = normals[idxs2[i]];
			double curr_area = all_face_area[idxs2[i]];
			double spatial_distance = dists[i];
			double g = GaussianWeight(spatial_distance, gaussian);
			var += g * curr_area * pow((mean_normal - nr_jk).length(), 2);
		}
		var /= normalizer;
		return var;
	}

	//detect
	double RollingGuidedNormalFiltering::getRollingMeanSqureAngleErro(const TriMesh &mesh, std::vector<TriMesh::Normal>& filtered_normals);
	void getMultipleRingFaceHandlerBaseOnVertex(TriMesh &mesh, TriMesh::VertexHandle vh, int ring, std::vector<TriMesh::FaceHandle> &face_neighbor);
	void updateVertexPosition(TriMesh &mesh, std::vector<TriMesh::Normal> &filtered_normals, int iteration_number, bool fixed_boundary);
	//conjugate discrend method
	void conjugateMethodUpdate(TriMesh & mesh, std::vector<TriMesh::Normal>& filtered_normals, int iternum);


	void getFaceNeigborhood(TriMesh &mesh, TriMesh::FaceHandle fh, double radius, std::vector<TriMesh::FaceHandle> &face_neighbor);
	void annSearchFaceNeighor(TriMesh::Point queryPt, double radius, std::vector<int>& idxs, std::vector<double>& dists);
private:
	ANNkd_tree*			kdTree_;
	ANNpointArray		dataPts_;

}; 

//GPU
//extern "C" void
//getData(TriMesh::Point *all_centriod, TriMesh::Normal *all_face_normal, double *all_face_area,
//	unsigned int num, TriMesh::Normal **filtered_ptr, double sigma_s, double sigma_r, int iter);
//extern "C" void
//updateVertexGlobal(std::vector <TriMesh::Point> all_centriod,
//	std::vector<TriMesh::Point> all_vertex, std::vector<double> all_face_area,
//	std::vector<TriMesh::Normal> filtered_normal,
//	TriMesh::Point **new_points, double sigma_s, double sigma_r);


#endif // ROLLINGGUIDEDNORMALFILTERING_H
