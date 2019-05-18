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
	void gpuUpdateFilteredNormals(TriMesh &mesh, std::vector<TriMesh::Normal> &filtered_normals);
	void gpuUpdateVertex(TriMesh &mesh, std::vector<TriMesh::Normal> &filtered_normals);



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


extern "C" void
getData(TriMesh::Point *all_centriod, TriMesh::Normal *all_face_normal, double *all_face_area,
	unsigned int num, TriMesh::Normal **filtered_ptr, double sigma_s, double sigma_r, int iter);
extern "C" void
updateVertexGlobal(std::vector <TriMesh::Point> all_centriod,
	std::vector<TriMesh::Point> all_vertex, std::vector<double> all_face_area,
	std::vector<TriMesh::Normal> filtered_normal,
	TriMesh::Point **new_points, double sigma_s, double sigma_r);


#endif // ROLLINGGUIDEDNORMALFILTERING_H
