#include "RollingGuidedNormalFiltering.h"
#include <QDebug>

//using in vs2015
#include <ctime>
#include <iostream>
#include <qmatrix.h>
#include <fstream>

void RollingGuidedNormalFiltering::getFaceNeigborhood(TriMesh & mesh, TriMesh::FaceHandle fh, double radius, std::vector<TriMesh::FaceHandle>& face_neighbor)
{
	
}

void RollingGuidedNormalFiltering::annSearchFaceNeighor(TriMesh::Point queryPt, double radius, std::vector<int>& idxs, std::vector<double>& dists)
{
	ANNdist dist = radius;
	int k_radius;
	//ANNdist radius = 1;
	// convert to ann point
	ANNpoint annPt;
	annPt = annAllocPt(3);
	annPt[0] = queryPt[0];
	annPt[1] = queryPt[1];
	annPt[2] = queryPt[2];



	// KNN search
	k_radius = kdTree_->annkFRSearch(
		annPt,
		radius,
		0);


	int k = k_radius;
	ANNidxArray			nnIdx;
	ANNdistArray		nndists;
	nnIdx = new ANNidx[k];
	nndists = new ANNdist[k];
	double eps = 0;

	k_radius = kdTree_->annkFRSearch(
		annPt,
		radius,
		k,
		nnIdx,
		nndists,
		eps);

	// write to nnIdx
	// write to nndists
	idxs.clear();
	dists.clear();
	for (int i = 0; i < k; i++) {
		idxs.push_back(nnIdx[i]);
		dists.push_back(nndists[i]);
	}
	if (nnIdx!=NULL)
		delete nnIdx;
	if (nndists != NULL)
		delete nndists;
}



RollingGuidedNormalFiltering::RollingGuidedNormalFiltering(DataManager *_data_manager, ParameterSet *_parameter_set)
    : MeshDenoisingBase(_data_manager, _parameter_set)
{
    initParameters();
	//testANN(data_manager_->getNoisyMesh());
	std::vector<TriMesh::Point> all_face_centroid;
	TriMesh mesh = data_manager_->getNoisyMesh();
	getFaceCentroid(mesh, all_face_centroid);

	int		nPts = all_face_centroid.size();			//顶点数
	dataPts_ = annAllocPts(nPts, 3);
	ANNpoint* dataPtsIt = dataPts_;
	//for(std::vector<TriMesh::Point>::iterator it = all_face_centroid.begin(); it!=all_face_centroid.end() ;it++)
	for (int i = 0; i<int(all_face_centroid.size()); i++)
	{
		TriMesh::Point pt = all_face_centroid.at(i);
		// assign x,y,z to record
		dataPts_[i][0] = pt[0];
		dataPts_[i][1] = pt[1];
		dataPts_[i][2] = pt[2];
	}

	kdTree_ = new ANNkd_tree(
		dataPts_,
		nPts,
		3);

	std::cout << "tree build finish" << std::endl;
}

RollingGuidedNormalFiltering::~RollingGuidedNormalFiltering()
{
	if (kdTree_ != NULL)
		delete kdTree_;
	if (dataPts_ != NULL)
		delete dataPts_;
}

void RollingGuidedNormalFiltering::denoise()
{

    clock_t start_time = clock();
 
    TriMesh mesh = data_manager_->getNoisyMesh();

    clock_t time_1 = clock();

	std::cout << "decompose time: " << static_cast<double>(time_1 - start_time) / CLOCKS_PER_SEC << " s" << std::endl;

    if(mesh.n_vertices() == 0)
        return;

    //update face normal
    std::vector<TriMesh::Normal> filtered_normal;
    //updateFilteredNormals(mesh, filtered_normal);

	//gpuUpdateFilteredNormals(mesh, filtered_normal);
	


	//data_manager_->getFilteredNormals(filtered_normal);
	//if (filtered_normal.empty())
	//{
		updateFilteredNormals(mesh, filtered_normal);
	//}



    clock_t time_2 = clock();
	std::cout << "filtering time : " << static_cast<double>(time_2 - time_1) / CLOCKS_PER_SEC << " s\n" << std::endl;
    
	//update vertex position
    int iter_num;
    parameter_set_->getValue(QString("Vectex iter_num"), iter_num);

	//RollingGuidedNormalFiltering::updateVertexPosition(mesh, filtered_normal, iter_num, false);
	//MeshDenoisingBase::updateVertexPosition(mesh, filtered_normal, iter_num, false);
	updateVertexPositionWithWeight(mesh, filtered_normal, iter_num, false);
	//updateVertexPosition(mesh, filtered_normal, 1, false, iter_num);
	//gpuUpdateVertex(mesh, filtered_normal);
	//conjugateMethodUpdate(mesh, filtered_normal, iter_num);
	clock_t end_time = clock();
	std::cout << "vertex update time: " << static_cast<double>(end_time - time_2) / CLOCKS_PER_SEC << " s\n" << std::endl;
    std::cout << "vertex update numbers: " << iter_num << std::endl;
	//update data
    data_manager_->setMesh(mesh);
    data_manager_->setDenoisedMesh(mesh);
	//for show filtered normals
	data_manager_->setFilteredNormal(filtered_normal);

	TriMesh filteredMesh = data_manager_->getDenoisedMesh();
	TriMesh originalMesh = data_manager_->getOriginalMesh();
	std::cout << "mean squre erro original: " << RollingGuidedNormalFiltering::getMeanSquareAngleError(filteredMesh, originalMesh) << std::endl;
	std::cout << "mean squre erro with filtered normal: " << RollingGuidedNormalFiltering::getRollingMeanSqureAngleErro(filteredMesh, filtered_normal) << std::endl;
	std::cout << "------------------------------------------------------------" << std::endl;
}

void RollingGuidedNormalFiltering::initParameters()
{
    parameter_set_->removeAllParameter();

    QStringList strList_DenoiseType;
    strList_DenoiseType.push_back(QString("Local"));
    strList_DenoiseType.push_back(QString("Global"));

    parameter_set_->addParameter(QString("Denoise Type"), strList_DenoiseType, 0, QString("Denoise Type"), QString("The type of denoise method."));

    QStringList strList_NeighborType;
    strList_NeighborType.push_back(QString("geometrical"));
    strList_NeighborType.push_back(QString("topological"));

    parameter_set_->addParameter(QString("Face Neighbor"), strList_NeighborType, 0, QString("Face Neighbor"), QString("The type of the neighbor of the face."));

    parameter_set_->addParameter(QString("Iteration Num."), 0, QString("Iteration Num."), QString("The total iteration number of the algorithm."),
                                 true, 0, 100);
    parameter_set_->addParameter(QString("Multiple(* sigma_s)"), 3.0, QString("Multiple(* sigma_s)"), QString("Standard deviation of spatial weight."),
                                 true, 1.0e-9, 100.0);
    parameter_set_->addParameter(QString("Multiple(* sigma_r)"), 0.2, QString("Multiple(* sigma_r)"), QString("Standard deviation of range weight."),
                                 true, 1.0e-9, 100.0);

    parameter_set_->addParameter(QString("Neighbor Size"), 3, QString("Neighbor Size"), QString("The size of neighbor based on radius or ring"));

	parameter_set_->addParameter(QString("Surf Size"), 1, QString("Surf Size"), QString("The size of surface change value"));
	parameter_set_->addParameter(QString("Var Size"), 1, QString("Var Size"), QString("The size of surface var value"));


	parameter_set_->addParameter(QString("Sigmoid Coeff"), 100.0, QString("Sigmoid Coeff"), QString("Sigmoid Coeff"),
		true, 1.0e-9, 10e8);

	parameter_set_->addParameter(QString("k Coeff"), 1.0, QString("k Coeff"), QString("k Coeff"),
		true, 1.0e-9, 10e8);
	parameter_set_->addParameter(QString("delta Coeff"), 1.0, QString("delta Coeff"), QString("delta Coeff"),
		true, 1.0e-9, 10e8);

	parameter_set_->addParameter(QString("Norm term"), 0.1, QString("Norm term"), QString("the norm term"),
		true, 1.0e-9, 100.0);
	parameter_set_->addParameter(QString("Threshold"), 0.0, QString("Threshold"), QString("Threshold"),
		true, 1.0e-9, 100.0);
	parameter_set_->addParameter(QString("LR"), 1.0, QString("LR"), QString("LR"),
		true, 1.0e-9, 100.0);
	parameter_set_->addParameter(QString("Vectex iter_num"), 200, QString("Vectex iter_num"), QString("The total iteration number of vecteice update."));


    parameter_set_->setName(QString("Rolling Guided Normal Filtering"));
    parameter_set_->setLabel(QString("Rolling Guided Normal Filtering"));
    parameter_set_->setIntroduction(QString("Rolling Guided Mesh Filtering -- Parameters"));

}
void RollingGuidedNormalFiltering::getVertexBasedFaceNeighbor(TriMesh &mesh, TriMesh::FaceHandle fh, std::vector<TriMesh::FaceHandle> &face_neighbor)
{
    getFaceNeighbor(mesh, fh, kVertexBased, face_neighbor);
}

void RollingGuidedNormalFiltering::getRadiusBasedFaceNeighbor(TriMesh &mesh, TriMesh::FaceHandle fh, double radius, std::vector<TriMesh::FaceHandle> &face_neighbor)
{
    TriMesh::Point ci = mesh.calc_face_centroid(fh);
    std::vector<bool> flag((int)mesh.n_faces(), false);

    face_neighbor.clear();
    flag[fh.idx()] = true;
    std::queue<TriMesh::FaceHandle> queue_face_handle;
    queue_face_handle.push(fh);

    std::vector<TriMesh::FaceHandle> temp_face_neighbor;

    while(!queue_face_handle.empty())
    {
        TriMesh::FaceHandle temp_face_handle_queue = queue_face_handle.front();
        if(temp_face_handle_queue != fh)
            face_neighbor.push_back(temp_face_handle_queue);
        queue_face_handle.pop();
        getVertexBasedFaceNeighbor(mesh, temp_face_handle_queue, temp_face_neighbor);
        for(int i = 0; i < (int)temp_face_neighbor.size(); i++)
        {
            TriMesh::FaceHandle temp_face_handle = temp_face_neighbor[i];
            if(!flag[temp_face_handle.idx()])
            {
                TriMesh::Point cj = mesh.calc_face_centroid(temp_face_handle);
                double distance = (ci - cj).length();
                if(distance <= radius)
                    queue_face_handle.push(temp_face_handle);
                flag[temp_face_handle.idx()] = true;
            }
        }
    }
}


void RollingGuidedNormalFiltering::getAllFaceNeighborGMNF(TriMesh &mesh, MeshDenoisingBase::FaceNeighborType face_neighbor_type, double radius, bool include_central_face,
                                                       std::vector<std::vector<TriMesh::FaceHandle> > &all_face_neighbor)
{
    std::vector<TriMesh::FaceHandle> face_neighbor;
    for(TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
    {
        if(face_neighbor_type == kVertexBased)
            getMultipleRingNeighbor(mesh, *f_it, radius, face_neighbor);
        else if(face_neighbor_type == kRadiusBased)
            getRadiusBasedFaceNeighbor(mesh, *f_it, radius, face_neighbor);
        if(include_central_face)
            face_neighbor.push_back(*f_it);
        all_face_neighbor[f_it->idx()] = face_neighbor;
    }
}

void RollingGuidedNormalFiltering::getGlobalMeanEdgeLength(TriMesh &mesh, double& mean_edge_length)
{
    int num = 0;

    double sum_edge_length = 0.0;
    for(TriMesh::EdgeIter e_it = mesh.edges_begin(); e_it != mesh.edges_end(); e_it++)
    {
        TriMesh::EdgeHandle eh = *e_it;
        sum_edge_length += mesh.calc_edge_length(eh);
        num++;
    }
    mean_edge_length = sum_edge_length / num;
}

void RollingGuidedNormalFiltering::getLocalMeanEdgeLength(TriMesh &mesh, std::vector<TriMesh::FaceHandle> &face_neighbor, double& local_mean_edge)
{
    std::vector<bool> edge_flag((int)mesh.n_edges(), false);
    double sum = 0.0;
    int num = 0;
    for(int i = 0; i < (int)face_neighbor.size(); i++)
    {
        for(TriMesh::FaceEdgeIter fe_it = mesh.fe_iter(face_neighbor[i]); fe_it.is_valid(); fe_it++)
        {
            //qDebug() << "edge_flag index: " << fe_it->idx();
            if(!edge_flag[fe_it->idx()])
            {
                edge_flag[fe_it->idx()] = true;
                sum += mesh.calc_edge_length(*fe_it);
                num++;
            }
        }
    }
    local_mean_edge = sum / num;
}



double RollingGuidedNormalFiltering::GaussianWeight(double Sqdistance, double sigma)
{
    return std::exp( -0.5 * Sqdistance / (sigma * sigma));
}

void RollingGuidedNormalFiltering::updateVertexPosition(TriMesh & mesh, std::vector<TriMesh::Normal>& filtered_normals, int iteration_number, bool fixed_boundary, unsigned int ring)
{
	std::vector<TriMesh::Point> new_points(mesh.n_vertices());

	std::vector<TriMesh::Point> centroid;
	std::vector<double> all_face_area;

	double sigma_S;
	double sigma_R;
	parameter_set_->getValue(QString("Multiple(* sigma_s)"), sigma_S);
	parameter_set_->getValue(QString("Multiple(* sigma_s)"), sigma_R);
	for (int iter = 0; iter < iteration_number; iter++)
	{
		getFaceCentroid(mesh, centroid);
		getFaceArea(mesh, all_face_area);
		//for every vertex
		for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
		{
			//current point
			TriMesh::Point p = mesh.point(*v_it);
			if (fixed_boundary && mesh.is_boundary(*v_it))
			{
				new_points.at(v_it->idx()) = p;
			}
			else
			{
				std::vector<TriMesh::FaceHandle> vertex_face_neighbor;
				getMultipleRingFaceHandlerBaseOnVertex(mesh, v_it, ring, vertex_face_neighbor);
				double mean_edge_len;
				getLocalMeanEdgeLength(mesh, vertex_face_neighbor, mean_edge_len);

				double total_weight = 0.0;
				TriMesh::Point temp_point(0.0, 0.0, 0.0);
				for (std::vector<TriMesh::FaceHandle>::iterator it = vertex_face_neighbor.begin(); it != vertex_face_neighbor.end(); ++it)
				{
					TriMesh::FaceHandle fh = TriMesh::FaceHandle(*it);
					TriMesh::Normal temp_normal = filtered_normals[it->idx()];
					TriMesh::Point temp_centroid = centroid[it->idx()];
					double spatital_distance = GaussianWeight((p - temp_centroid).length(), sigma_S * mean_edge_len);
					double range_diistance = GaussianWeight(((temp_normal | (temp_centroid - p))*temp_normal).length(), sigma_R);
					double area = all_face_area[it->idx()];
					double temp_weight = spatital_distance * range_diistance * area;
					total_weight += temp_weight;

					temp_point += temp_weight * temp_normal * (temp_normal | (temp_centroid - p));
				}
				p += temp_point / total_weight;
				new_points.at(v_it->idx()) = p;
			}
		}

		for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
		{
			mesh.set_point(*v_it, new_points[v_it->idx()]);
		}
	}
}

void RollingGuidedNormalFiltering::getMultipleRingNeighbor(TriMesh &mesh, TriMesh::FaceHandle fh, int n, std::vector<TriMesh::FaceHandle> &face_neighbor)
{
	
    face_neighbor.clear();
    std::set<int> neighbor_face_index; neighbor_face_index.clear();

    std::set<int> temp_face_neighbor_index;
    temp_face_neighbor_index.clear();
    temp_face_neighbor_index.insert(fh.idx());
    for (int i = 0; i < n; i++)
    {
        for (std::set<int>::iterator iter = temp_face_neighbor_index.begin(); iter != temp_face_neighbor_index.end(); ++iter)
        {
            TriMesh::FaceHandle fh_j = TriMesh::FaceHandle(*iter);
            for(TriMesh::FaceVertexIter fv_it = mesh.fv_begin(fh_j); fv_it.is_valid(); fv_it++)
            {
                for(TriMesh::VertexFaceIter vf_it = mesh.vf_iter(*fv_it); vf_it.is_valid(); vf_it++)
                    if((*vf_it) != fh)
                        neighbor_face_index.insert(vf_it->idx());
            }
        }

        ///int debug_size = temp_face_neighbor_index.size();

        temp_face_neighbor_index.clear();
        temp_face_neighbor_index = neighbor_face_index;

        //update temp_face_neighbor_index
        /*std::set_difference(neighbor_face_index.begin(), neighbor_face_index.end(),
                            temp_face_neighbor_index.begin(), temp_face_neighbor_index.end(),
                            std::insert_iterator<set<int> >(temp_face_neighbor_index,temp_face_neighbor_index.begin()));*/
    }

    //update  face_neighbor
    for(std::set<int>::iterator iter = neighbor_face_index.begin(); iter != neighbor_face_index.end(); ++iter)
    {
        face_neighbor.push_back(TriMesh::FaceHandle(*iter));
    }
}

void RollingGuidedNormalFiltering::updateVertexPositionWithWeight(TriMesh & mesh, std::vector<TriMesh::Normal>& filtered_normals, int iteration_number, bool fixed_boundary)
{
	double lambda = 0.0;
	parameter_set_->getValue(QString("Norm term"), lambda);
	double threshold = 0.0;
	parameter_set_->getValue(QString("Threshold"), threshold);
	qDebug() << "lambda : " << lambda;
	qDebug() << "threshold : " << threshold;
	double sigma = 0.0;



	std::vector<TriMesh::Point> new_points(mesh.n_vertices());
	std::vector<TriMesh::Point> centroid;
	std::vector<TriMesh::Normal> all_face_normal;
	getFaceNormal(mesh, all_face_normal);
	std::vector<double> all_face_area;
	getFaceArea(mesh, all_face_area);
	std::vector<double> point_weights(mesh.n_faces());

	// get point constraint weights
	//std::vector<std::vector<TriMesh::FaceHandle>> all_face_neighbors;
	//MeshDenoisingBase::getAllFaceNeighbor(mesh, all_face_neighbors);
	double var_min = +10e8;
	double var_max = -10e8;
	double change_min = +10e8;
	double change_max = -10e8;

	int neighbor_size_factor;
	int var_neighbor_size;
	int surf_neighbor_size;
	double k;
	double sigmoid_coeff;
	double lr = 0.0;
	double delta;
	if (!parameter_set_->getValue(QString("Neighbor Size"), neighbor_size_factor));
	if (!parameter_set_->getValue(QString("Surf Size"), surf_neighbor_size));
	if (!parameter_set_->getValue(QString("Var Size"), var_neighbor_size));
	if (!parameter_set_->getValue(QString("Sigmoid Coeff"), sigmoid_coeff));
	if (!parameter_set_->getValue(QString("k Coeff"), k));
	if (!parameter_set_->getValue(QString("LR"), lr));
	if (!parameter_set_->getValue(QString("delta Coeff"), delta));

	qDebug() << "k coeff" << k;
	qDebug() << "sigmoid coeff" << sigmoid_coeff;
	qDebug() << "LR " << lr;
	qDebug() << "delta " << delta;
	double mean_edge_length;
	getGlobalMeanEdgeLength(mesh, mean_edge_length);
	double radius = mean_edge_length * surf_neighbor_size;
	double SsqRad = radius * radius;
	double VsqRad = pow((mean_edge_length * var_neighbor_size), 2);
	std::vector<TriMesh::Point> face_centriod;
	getFaceCentroid(mesh, face_centriod);



	std::vector<double> change_list(mesh.n_faces());
	std::vector<double> var_list(mesh.n_faces());
	std::vector<TriMesh::Normal> gradient_normal;
	std::vector<TriMesh::Normal> filtered_gradient_normal;
	getGradientNormal(mesh, all_face_normal, gradient_normal);
	getGradientNormal(mesh, filtered_normals, filtered_gradient_normal);

	for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
	{
		double point_constrain = 0.0;
		double neighbor_area = 0.0;;
		// get neighbor
		TriMesh::Point ci = face_centriod.at(f_it->idx());
		std::vector<int> idxs; // all_face_neighbor_idx[index_i];
		std::vector<double> dists; // all_face_neighbor_dist[index_i];
		annSearchFaceNeighor(ci, SsqRad, idxs, dists);
		std::vector<int> idxs2; // all_face_neighbor_idx[index_i];
		std::vector<double> dists2; // all_face_neighbor_dist[index_i];
		annSearchFaceNeighor(ci, VsqRad, idxs2, dists2);

		TriMesh::Normal no_i = all_face_normal[f_it->idx()];
		// lambda_1
		TriMesh::Normal rtv = TriMesh::Normal(0.0, 0.0, 0.0);
		double area_sum = 0.0;
		double x=0.0, y=0.0, z=0.0;
		for (int i = 0; i < int(idxs.size()); i++)
		{
			//TriMesh::Normal no_jk = all_face_normal[idxs[i]];
			//TriMesh::Normal nr_jk = filtered_normals[idxs[i]];
			//double curr_area = all_face_area[idxs[i]];
			//point_constrain += curr_area * (pow(((no_jk - nr_jk).length()), 2));
			//area_sum += curr_area;
			double spatial_distance = dists[i];
			//printf("(%d,%d)", spatial_distance, spatial_distances);
			double g = GaussianWeight(spatial_distance, mean_edge_length * 3);
			TriMesh::Normal no_jk = gradient_normal[idxs[i]];// -no_i;
			rtv += g * no_jk;
			x += g * pow(abs(no_jk[0]),2);
			y += g * pow(abs(no_jk[1]),2);
			z += g * pow(abs(no_jk[2]),2);
		}
		double epsilon = 10e-8;
		point_constrain = x + y + z;
		//point_constrain = pow(x,2) + pow(y,2) + pow(z,2);//x / (abs(rtv[0]) + epsilon) + y / (abs(rtv[1]) + epsilon) + z / (abs(rtv[2]) + epsilon);
		point_constrain /= int(idxs.size());

		// lambda_2
		double point_large = 0.0;
		TriMesh::Normal mean_normal = TriMesh::Normal(0.0, 0.0, 0.0);
		double normalizer = 0.0;
		double var = 0.0;
		var = getCovariance(idxs2, dists2, all_face_area, filtered_normals, mean_edge_length * 3);
		//double xx = 0.0, yy = 0.0, zz = 0.0;
		//for (int i = 0; i < int(idxs2.size()); i++)
		//{
		//	//TriMesh::Normal no_jk = all_face_normal[idxs[i]];
		//	//TriMesh::Normal nr_jk = filtered_normals[idxs[i]];
		//	//double curr_area = all_face_area[idxs[i]];
		//	//point_constrain += curr_area * (pow(((no_jk - nr_jk).length()), 2));
		//	//area_sum += curr_area;
		//	double spatial_distance = dists2[i];
		//	//printf("(%d,%d)", spatial_distance, spatial_distances);
		//	double g = GaussianWeight(spatial_distance, mean_edge_length * 3);
		//	TriMesh::Normal no_jk = filtered_gradient_normal[idxs2[i]];// -no_i;
		//	rtv += g * no_jk;
		//	xx += g * pow(abs(no_jk[0]), 2);
		//	yy += g * pow(abs(no_jk[1]), 2);
		//	zz += g * pow(abs(no_jk[2]), 2);
		//}
		//var = xx + yy + zz;
		//var /= int(idxs2.size());

		//area_sum = 0.0;
		//for (int i = 0; i < int(idxs2.size()); i++)
		//{
		//	double curr_area = all_face_area[idxs2[i]];
		//	TriMesh::Normal nr_jk = filtered_normals[idxs2[i]];
		//	mean_normal += curr_area * nr_jk;
		//	//normalizer += curr_area * nr_jk.length();
		//}
		//mean_normal = mean_normal.normalize();
		//if (mean_normal.length() - 1 > 10e-5)
		//	qDebug() << "mean_normal len: " << mean_normal.length();
		//for (int i = 0; i < int(idxs2.size()); i++)
		//{
		//	TriMesh::Normal nr_jk = filtered_normals[idxs2[i]];
		//	double curr_area = all_face_area[idxs2[i]];
		//	double spatial_distance = dists2[i];
		//	//printf("(%d,%d)", spatial_distance, spatial_distances);
		//	double g = GaussianWeight(spatial_distance, mean_edge_length * 3);
		//	//var += curr_area * pow((filtered_normals[f_it->idx()] - nr_jk).length(), 2);
		//	var += pow((mean_normal - nr_jk).length(), 2);
		//	//area_sum += curr_area;
		//}
		//var /= (idxs2.size() - 1);

		var_list.at(f_it->idx()) = var;
		//abs(gradient_normal[f_it->idx()][0]) + abs(gradient_normal[f_it->idx()][1]) + abs(gradient_normal[f_it->idx()][2]);
		// var_list.at(f_it->idx()) = var;
		change_list.at(f_it->idx()) = point_constrain;

		// compute min max
		if (var_max < var)
			var_max = var;
		if (var_min > var)
			var_min = var;
		if (change_max < point_constrain)
			change_max = point_constrain;
		if (change_min > point_constrain)
			change_min = point_constrain;

		point_weights.at(f_it->idx()) = point_constrain;
		
	}

	std::ofstream changeTxt("change.txt");
	std::ofstream varTxt("var.txt");
	for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
	{
		changeTxt << change_list[f_it->idx()] << ",";
		varTxt << var_list[f_it->idx()] << ",";
	}
	changeTxt.close();
	varTxt.close();

	std::vector<double> values(mesh.n_faces());
	double change_range = change_max - change_min;
	double var_range = var_max - var_min;
	
	qDebug() << "change min " << change_min << "change max " << change_max;
	qDebug() << "variant min " << var_min << "variant max " << var_max;
	qDebug() << "change range " << change_range << " var_range" << var_range;

	double max = -10e8;
	double min = +10e8;

	for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++) {

		double lambda_1 = (change_list.at(f_it->idx())); // *10 / change_range;
		double lambda_2 = (var_list.at(f_it->idx()));	// *10 / var_range;
		//double response = sigmoid_coeff * (delta * pow(lambda_2, 2) - k * pow(lambda_1, 2));
		double response = sigmoid_coeff * (delta * lambda_2 - (k * lambda_1));
		values.at(f_it->idx()) = 1 / (1 + exp(-response));
		//values.at(f_it->idx()) = change_list.at(f_it->idx());
		if (max < response)
			max = response;
		if (min > response)
			min = response;

		if (values.at(f_it->idx()) < threshold)
		{
			point_weights.at(f_it->idx()) = 0;
		}
		else {
			point_weights.at(f_it->idx()) = values.at(f_it->idx());
		}
	}


	qDebug() << "response max " << max << " response min" << min;

	data_manager_->setRenderValues(values);
	std::vector<TriMesh::Point> old_points(mesh.n_vertices());
	for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
		old_points.at(v_it->idx()) = mesh.point(*v_it);
		

	// gradient descent
	for (int iter = 0; iter < iteration_number; iter++)
	{
		getFaceCentroid(mesh, centroid);
		for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
		{
			TriMesh::Point p = mesh.point(*v_it);
			if (fixed_boundary && mesh.is_boundary(*v_it))
			{
				new_points.at(v_it->idx()) = p;
			}
			else
			{
				//double temp_face_area = 0.0;
				double face_num = 0;
				double point_penels = 0.0;
				double weight_sum = 0.0;
				TriMesh::Point temp_point(0.0, 0.0, 0.0);
				for (TriMesh::VertexFaceIter vf_it = mesh.vf_iter(*v_it); vf_it.is_valid(); vf_it++)
				{
					TriMesh::Normal temp_normal = filtered_normals[vf_it->idx()];
					TriMesh::Point temp_centroid = centroid[vf_it->idx()];
					//double curr_area = all_face_area[vf_it->idx()];
					//double curr_penel = point_weights[vf_it->idx()];


					double pWeight = point_weights[vf_it->idx()];
					weight_sum += pWeight;
					//temp_face_area += curr_area;
					//point_penels += curr_penel;
					//point_penels += /*curr_penel*/  curr_area;
					temp_point += /*pWeight */ temp_normal * (temp_normal | (temp_centroid - p));
					//normalize_term += pWeight;
					face_num++;
					
				}
				TriMesh::Point old_point = old_points.at(v_it->idx());
				p += lr * lambda * weight_sum * (old_point - p) / (face_num);;
				p += lr * temp_point / face_num;//(normalize_term);

				new_points.at(v_it->idx()) = p;
			}
		}



		// update vertex
		for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
			mesh.set_point(*v_it, new_points[v_it->idx()]);

		// output energy
		double energy = 0.0;
		double regularizer = 0.0;
		for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++) 
		{
			double pWeight = point_weights[f_it->idx()];
			TriMesh::Point ci = centroid[f_it->idx()];
			for (TriMesh::FaceVertexIter fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); fv_it++)
			{
				energy += powl(all_face_normal[fv_it->idx()] | (ci - mesh.point(*fv_it)), 2);
				TriMesh::Point old_point = old_points.at(fv_it->idx());
				regularizer +=  pWeight * powl((mesh.point(*fv_it) - old_point).length(), 2);
			}
		}
		if (iter % 10 == 0) 
		{
			qDebug() << "Step " << iter << " Loss " << energy + lambda * regularizer << " Energy "
				<< energy << " Regularize:" << lambda * regularizer; /// mesh.n_faces();
		}
	}
}

void RollingGuidedNormalFiltering::updateFilteredNormals(TriMesh &mesh, std::vector<TriMesh::Normal> &filtered_normals)
{
    // get parameter for normal update
    int denoise_index;
    if(!parameter_set_->getStringListIndex(QString("Denoise Type"), denoise_index))
        return;

    DenoiseType denoise_type = (denoise_index == 0) ? kLocal : kGlobal;

    if(denoise_type == kLocal)
        updateFilteredNormalsLocalScheme(mesh, filtered_normals);
    else if (denoise_type == kGlobal)
        updateFilteredNormalsGlobalScheme(mesh, filtered_normals);
}

void RollingGuidedNormalFiltering::updateFilteredNormalsLocalScheme(TriMesh &mesh, std::vector<TriMesh::Normal> &filtered_normals)
{
	//initialize
	//getFaceNormal(mesh, filtered_normals);
	filtered_normals = std::vector<TriMesh::Normal>(mesh.n_faces(), TriMesh::Normal(0.0, 0.0, 0.0));
	//get parameter
	int iteration_number;
	if (!parameter_set_->getValue(QString("Iteration Num."), iteration_number))
		return;
	//if (iteration_number == 0)
	//{
	//	getFaceNormal(data_manager_->getOriginalMesh(), filtered_normals);
	//	return;
	//}

	double sigma_s;
	if (!parameter_set_->getValue(QString("Multiple(* sigma_s)"), sigma_s))
		return;
	double sigma_r;
	if (!parameter_set_->getValue(QString("Multiple(* sigma_r)"), sigma_r))
		return;

	int neighbor_size_factor;
	if (!parameter_set_->getValue(QString("Neighbor Size"), neighbor_size_factor));

	int face_neighbor_index;
	if (!parameter_set_->getStringListIndex(QString("Face Neighbor"), face_neighbor_index))
		return;

	FaceNeighborType face_neighbor_type = face_neighbor_index == 0 ? kRadiusBased : kVertexBased;

	std::vector<double> all_face_area;
	getFaceArea(mesh, all_face_area);

	std::vector<TriMesh::Normal> all_face_normal;
	getFaceNormal(mesh, all_face_normal);

	std::vector<TriMesh::Point> face_centriod;
	getFaceCentroid(mesh, face_centriod);

	double mean_edge_length;
	getGlobalMeanEdgeLength(mesh, mean_edge_length);
	double radius = mean_edge_length * neighbor_size_factor;

	for (int k = 0; k < iteration_number; k++)
	{
		std::vector<double> tempature(mesh.n_faces(), 0.0);
		std::vector<TriMesh::Normal> temp_normals = std::vector<TriMesh::Normal>(mesh.n_faces(), TriMesh::Normal(0.0, 0.0, 0.0));
		std::vector<TriMesh::Normal> gradient_normal;
		getGradientNormal(mesh, filtered_normals, gradient_normal);
		for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
		{

			int index_i = f_it->idx();
			TriMesh::Normal ni_k = filtered_normals[index_i];
			TriMesh::Normal n_temp(0.0, 0.0, 0.0);
			TriMesh::Point ci = face_centriod[index_i];

			// get neighbor
			std::vector<int> idxs; // all_face_neighbor_idx[index_i];
			std::vector<double> dists; // all_face_neighbor_dist[index_i];
			double sqRad = radius * radius;
			annSearchFaceNeighor(ci, sqRad, idxs, dists);

			// get sigma tempature
			//TriMesh::Normal rtv = TriMesh::Normal(0.0, 0.0, 0.0);
			//double x = 0.0, y = 0.0, z = 0.0;
			//for (int j = 0; j < (int)idxs.size(); j++)
			//{
			//	double spatial_distance = dists[j];
			//	//printf("(%d,%d)", spatial_distance, spatial_distances);
			//	double g = GaussianWeight(spatial_distance, mean_edge_length * 3);
			//	TriMesh::Normal no_jk = gradient_normal[idxs[j]];// -no_i;
			//	rtv += g * no_jk;
			//	x += g * pow(abs(no_jk[0]), 2);
			//	y += g * pow(abs(no_jk[1]), 2);
			//	z += g * pow(abs(no_jk[2]), 2);
			//}

			//tempature.at(f_it->idx()) = (x + y + z) / idxs.size();

			// compute ni_k
			double weight_sum = 0.0;
			// get le
			// getLocalMeanEdgeLength(mesh, face_neighbor, mean_edge_length);
			for (int j = 0; j < (int)idxs.size(); j++)
			{
				int index_j = idxs[j];
				TriMesh::Normal nj_k = filtered_normals[index_j];
				TriMesh::Normal nj = all_face_normal[index_j];
				TriMesh::Point cj = face_centriod[index_j];
				//double spatial_distance = ((ci - cj).length())*(ci - cj).length();
				double spatial_distance = dists[j];
				//printf("(%d,%d)", spatial_distance, spatial_distances);
				double spatial_weight = GaussianWeight(spatial_distance, sigma_s * mean_edge_length);
				double range_distance = (ni_k - nj).length() * (ni_k - nj).length();
				//double range_distance = 0.0;
				double range_weight = GaussianWeight(range_distance, sigma_r);
				double face_area = all_face_area[index_j];
				double weight = face_area * range_weight * spatial_weight;
				weight_sum += weight;
				n_temp += weight * nj;
			}
			// update temp_normals
			n_temp /= weight_sum;
		    n_temp.normalize();
			temp_normals[index_i] = n_temp;
		}
			//update filtered_normals for iteration
		filtered_normals = temp_normals;
	}
		//for show
	data_manager_->setFilteredNormal(filtered_normals);
}


void RollingGuidedNormalFiltering::updateFilteredNormalsGlobalScheme(TriMesh &mesh, std::vector<TriMesh::Normal> &filtered_normals)
{
    //initialize
    filtered_normals = std::vector<TriMesh::Normal>(mesh.n_faces(), TriMesh::Normal(0.0, 0.0, 0.0));
    //get parameter
    int iteration_number;
    if(!parameter_set_->getValue(QString("Iteration Num."), iteration_number))
        return;
    double sigma_s;
    if(!parameter_set_->getValue(QString("Multiple(* sigma_s)"), sigma_s))
        return;
    double sigma_r;
    if(!parameter_set_->getValue(QString("Multiple(* sigma_r)"), sigma_r))
        return;

    std::vector<double> all_face_area;
    getFaceArea(mesh, all_face_area);

    std::vector<TriMesh::Normal> all_face_normal;
    getFaceNormal(mesh, all_face_normal);

    std::vector<TriMesh::Point> face_centriod;
    getFaceCentroid(mesh, face_centriod);

    double mean_edge_length = 0.0;
    getGlobalMeanEdgeLength(mesh, mean_edge_length);

    for(int k = 0; k < iteration_number; k++)
    {
        std::vector<TriMesh::Normal> temp_normals = std::vector<TriMesh::Normal>(mesh.n_faces(), TriMesh::Normal(0.0, 0.0, 0.0));
        for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
        {
            //get neigbor F
            int index_i = f_it->idx();
            TriMesh::Normal ni_k = filtered_normals[index_i];
            TriMesh::Normal n_temp(0.0, 0.0, 0.0);
            TriMesh::Point ci = face_centriod[index_i];
            //compute ni_k
            double weight_sum = 0.0;
            //get all face
            for(TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
            {
                int j = f_it->idx();
                TriMesh::Normal nj_k = filtered_normals[j];
                TriMesh::Normal nj = all_face_normal[j];
                TriMesh::Point cj = face_centriod[j];
                double spatial_distance = (ci - cj).length();
                double spatial_weight = GaussianWeight(spatial_distance, sigma_s * mean_edge_length);
                double range_distance = (ni_k - nj_k).length();
                //double range_distance = 0.0;
                double range_weight = GaussianWeight(range_distance, sigma_r);
                double face_area = all_face_area[j];
                double weight = face_area * range_weight * spatial_weight;
                weight_sum += weight;
                n_temp += weight * nj;
            }
            //update temp_normals
            n_temp /= weight_sum;
            n_temp.normalize();
            temp_normals[index_i] = n_temp;
        }
        //update filtered_normals for iteration
        filtered_normals = temp_normals;
    }
}

/*this function launch the GPU acceration
frist it copy the data needed to device
then get result and write back to host */
//void RollingGuidedNormalFiltering::gpuUpdateFilteredNormals(TriMesh &mesh, std::vector<TriMesh::Normal> &filtered_normals)
//{
//	//get parameter
//	int iteration_number;
//	if (!parameter_set_->getValue(QString("Iteration Num."), iteration_number))
//		return;
//	double sigma_s;
//	if (!parameter_set_->getValue(QString("Multiple(* sigma_s)"), sigma_s))
//		return;
//	double sigma_r;
//	if (!parameter_set_->getValue(QString("Multiple(* sigma_r)"), sigma_r))
//		return;
//
//	/*TO TEST THE CUDA CALL HERE*/
//	std::vector<double> all_face_area;
//	getFaceArea(mesh, all_face_area);
//
//	std::vector<TriMesh::Normal> all_face_normal;
//	getFaceNormal(mesh, all_face_normal);
//
//	std::vector<TriMesh::Point> face_centriod;
//	getFaceCentroid(mesh, face_centriod);
//
//	TriMesh::Normal *filtered_normals_ptr = NULL;
//
//	double mean_edge_length = 0.0;
//	getGlobalMeanEdgeLength(mesh, mean_edge_length);
//	//pass the data to from host to device
//	getData(&face_centriod[0], &all_face_normal[0], &all_face_area[0], mesh.n_faces(), 
//		&filtered_normals_ptr, (sigma_s * mean_edge_length), sigma_r, iteration_number);
//
//	if (filtered_normals_ptr == NULL)
//	{
//		//something erro happend here
//		std::cout << "ptr is null erro" << std::endl;
//	}
//	else {
//
//		//copy array filtered_normal form device to host
//		std::vector<TriMesh::Normal> _filtered_normals;
//		filtered_normals.reserve(mesh.n_faces());
//		filtered_normals.resize(mesh.n_faces());
//		filtered_normals.assign(&filtered_normals_ptr[0], &filtered_normals_ptr[mesh.n_faces()]);
//		data_manager_->setFilteredNormal(filtered_normals);
//	}
//	/*END TEST THE CUDA CALL HERE*/
//}
//
//void RollingGuidedNormalFiltering::gpuUpdateVertex(TriMesh & mesh, std::vector<TriMesh::Normal>& filtered_normals)
//{
//	std::vector<TriMesh::Point> all_centroid;
//	getFaceCentroid(mesh, all_centroid);
//	std::vector<TriMesh::Point> all_vertex(mesh.n_vertices(), TriMesh::Point(0.0, 0.0, 0.0));
//	for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
//	{
//		all_vertex.at(v_it->idx()) = mesh.point(v_it);
//	}
//
//	std::vector<double> all_face_area;
//	getFaceArea(mesh, all_face_area);
//
//	TriMesh::Point *new_points = NULL;
//	//注意 new_point 的值不能为空
//	double sigma_s;
//	double sigma_r;
//	double mean_edge_len;
//	getGlobalMeanEdgeLength(mesh, mean_edge_len);
//	parameter_set_->getValue(QString("Multiple(* sigma_s)"), sigma_s);
//	parameter_set_->getValue(QString("Multiple(* sigma_r)"), sigma_r);
//
//	//updateVertexGlobal(all_centroid, all_vertex, all_face_area, filtered_normals, &new_points, sigma_s * mean_edge_len, sigma_r);
//
//	for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
//	{
//		mesh.set_point(v_it, new_points[v_it->idx()]);
//	}
//}

double RollingGuidedNormalFiltering::getRollingMeanSqureAngleErro(const TriMesh &mesh, std::vector<TriMesh::Normal>& filtered_normals)
{
	
	
	double mean_square_angle_error = 0.0;
	std::vector<TriMesh::Normal>::iterator f_it2 = filtered_normals.begin();
	for (TriMesh::FaceIter f_it1 = mesh.faces_begin();
		f_it1 != mesh.faces_end(); f_it1++, f_it2++)
	{
		TriMesh::Normal normal1 = mesh.normal(*f_it1);
		TriMesh::Normal normal2 = *f_it2;
		double cross_value = normal1 | normal2;
		cross_value = std::min(1.0, std::max(cross_value, -1.0));
		mean_square_angle_error += std::acos(cross_value) * 180.0 / M_PI;
	}

	return mean_square_angle_error / (double)mesh.n_faces();
}

void RollingGuidedNormalFiltering::getMultipleRingFaceHandlerBaseOnVertex(TriMesh & mesh, TriMesh::VertexHandle vh, int ring, std::vector<TriMesh::FaceHandle>& face_neighbor)
{
	std::queue<TriMesh::FaceHandle> queue_face_handle;
	std::vector<bool> flag(mesh.n_faces(), false);

	//initial
	for (TriMesh::VertexFaceIter vf_it = mesh.vf_begin(vh); vf_it.is_valid(); ++vf_it)
	{
		queue_face_handle.push(vf_it);
		face_neighbor.push_back(vf_it);
		flag[vf_it->idx()] = true;
	}
	
	for (int i = 1; i < ring; ++i)
	{
		//Do some thing here
		int count = 1;
		int size = queue_face_handle.size();
		while (!queue_face_handle.empty())
		{
			if (count > size) break;
			TriMesh::FaceHandle currfh = queue_face_handle.front();
			queue_face_handle.pop();
			std::vector<TriMesh::FaceHandle> temp_face_neighbor;
			getFaceNeighbor(mesh, currfh, kVertexBased, temp_face_neighbor);
			for (std::vector<TriMesh::FaceHandle>::iterator fh_it = temp_face_neighbor.begin(); fh_it != temp_face_neighbor.end(); ++fh_it)
			{
				if (!flag[fh_it->idx()])
				{
					flag[fh_it->idx()] = true;
					queue_face_handle.push(*fh_it);
					face_neighbor.push_back(*fh_it);
				}
			}

			count++;
		}
	}
}

void RollingGuidedNormalFiltering::updateVertexPosition(TriMesh & mesh, std::vector<TriMesh::Normal>& filtered_normals, int iteration_number, bool fixed_boundary)
{
	std::vector<TriMesh::Point> new_points(mesh.n_vertices());

	std::vector<TriMesh::Point> centroid;

	std::vector<double> all_face_area;
	getFaceArea(mesh, all_face_area);
	for (int iter = 0; iter < iteration_number; iter++)
	{
		getFaceCentroid(mesh, centroid);
		for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
		{
			TriMesh::Point p = mesh.point(*v_it);
			if (fixed_boundary && mesh.is_boundary(*v_it))
			{
				new_points.at(v_it->idx()) = p;
			}
			else
			{
				double temp_face_area = 0.0;
				TriMesh::Point temp_point(0.0, 0.0, 0.0);
				for (TriMesh::VertexFaceIter vf_it = mesh.vf_iter(*v_it); vf_it.is_valid(); vf_it++)
				{
					TriMesh::Normal temp_normal = filtered_normals[vf_it->idx()];
					TriMesh::Point temp_centroid = centroid[vf_it->idx()];
					double curr_area = all_face_area[vf_it->idx()];
					temp_point += curr_area * temp_normal * (temp_normal | (temp_centroid - p));
					
					temp_face_area += curr_area;
				}
				p += temp_point / temp_face_area;

				new_points.at(v_it->idx()) = p;
			}
		}

		for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
			mesh.set_point(*v_it, new_points[v_it->idx()]);
	}
}

void RollingGuidedNormalFiltering::conjugateMethodUpdate(TriMesh & mesh, std::vector<TriMesh::Normal>& filtered_normals, int iternum)
{
	double lenth = 0.0;
	//initialize resiual and basis
	int n = mesh.n_vertices();
	std::vector<TriMesh::Point> r(n);
	std::vector<TriMesh::Point> basis(n);
	std::vector<TriMesh::Point> c;
	getFaceCentroid(mesh, c);
	std::vector<TriMesh::Normal> normal = filtered_normals;
	//getFaceNormal(mesh, normal);
	for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
	{
		int i = v_it->idx();
		TriMesh::Point p = mesh.point(*v_it);
		TriMesh::Normal ngd = TriMesh::Normal(0.0, 0.0, 0.0);
		//compute negative gradient 
		for (TriMesh::VertexFaceIter vf_it = mesh.vf_iter(*v_it); vf_it.is_valid(); ++vf_it)
		{
			int k = vf_it->idx();
			auto temp = (c[k] - p);
			auto curr_normal = normal[k];
			auto tempweight = curr_normal | temp;
			long double anothervalue = 0.0;
			long double a = curr_normal[0] * temp[0];
			long double b = curr_normal[1] * temp[1];
			long double c2 = curr_normal[2] * temp[2];
			long double wtf = a + b + c2;
			anothervalue = curr_normal[0] *  temp[0] + curr_normal[1] * temp[1] + curr_normal[2] * temp[2];
			ngd += normal[k] * (normal[k] | (c[k] - p)) * 3;
		}

		r[i] = ngd;
		basis[i] = r[i];
	}

	int k = 0;
	while (k++ < n)
	{
		//compute ak
		double ak = 0.0;
		double numitor = 0.0;
		double denumitor = 0.0;
		for (int i = 0; i < n; i++)
		{
			numitor += r[i] | r[i];
		}
		for (TriMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it)
		{
			double Fk = 0.0;
			std::vector<int> index;
			index.clear();
			for (TriMesh::FaceVertexIter fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it)
			{
				index.push_back(fv_it->idx());
			}
			for (int i = 0; i < index.size(); ++i)
			{
				Fk += normal[f_it->idx()] | (basis[index[i % index.size()]] - basis[index[(i - 1) % index.size()]]);
			}
			denumitor += Fk * Fk;
		}
		ak = numitor / denumitor;

		//update vertex;
		std::vector<TriMesh::Point> new_point(n);
		std::vector<TriMesh::Point> new_r(n);
		//compute Xk+1, Rk+1
		for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
		{
			//compute new x
			auto x = mesh.point(v_it);
			new_point[v_it->idx()] = x + ak * basis[v_it->idx()];

			//compute new r
			auto pb = basis[v_it->idx()];
			//calculate basis centoid
			TriMesh::Point centroid(0.0, 0.0, 0.0);
			for (TriMesh::VertexFaceIter vf_it = mesh.vf_iter(*v_it); vf_it.is_valid(); ++vf_it)
			{
				std::vector<int> index;
				for (TriMesh::FaceVertexIter fv_it = mesh.fv_iter(vf_it); fv_it.is_valid(); ++fv_it)
				{
					index.push_back(fv_it->idx());
				}
				for (int i = 0; i < index.size(); ++i)
				{
					centroid += basis[i];
				}
			}
			//compute akApk
			TriMesh::Point Apk(0.0, 0.0, 0.0);
			for (TriMesh::VertexFaceIter vf_it = mesh.vf_iter(*v_it); vf_it.is_valid(); ++vf_it)
			{
				int k = vf_it->idx();
				Apk += normal[k] * (normal[k] | (pb - centroid)) * 3;
			}
			new_r[v_it->idx()] = r[v_it->idx()] - ak * Apk;
		}

		//update vertex
		for (TriMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it)
		{
			mesh.set_point(v_it, new_point[v_it->idx()]);
		}


		//if rk+1 is suffiently small then exit loop
		double value = 0.0;
		for (std::vector<TriMesh::Point>::iterator it = new_r.begin(); it != new_r.end(); ++it)
		{
			value += (*it | *it);
		}
		lenth = value;
		//if (lenth < 10e-10);
		//calculate beta
		double rsold = 0.0;
		double rsnew = 0.0;
		for (int i = 0; i < n; i++)
		{
			rsold += r[i] | r[i];
			rsnew += new_r[i] | new_r[i];
		}

		for (int i = 0; i < n; i++)
		{
			basis[i] = basis[i] + (rsnew / rsold) * basis[i];
			r[i] = new_r[i];
		}

		if (k > iternum) break;
	}

	std::cout << "rk+1 value:" << lenth << endl;
}





