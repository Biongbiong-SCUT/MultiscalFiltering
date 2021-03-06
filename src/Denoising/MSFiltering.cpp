#include<MSFiltering.h>
#include <qdebug.h>

MSFiltering::MSFiltering(DataManager * _data_manager, ParameterSet * _parameter_set)
	:MeshDenoisingBase(_data_manager, _parameter_set), linear_solver_(Parameters::LDLT), system_matrix_factorized_(false)
{
	initParameters();
	mesh_ = data_manager_->getNoisyMesh();
	initKDtree();
}

MSFiltering::~MSFiltering()
{
	if (dataPts_ != NULL)
	{
		delete dataPts_;
		dataPts_ = NULL;
	}
	if (kdTree_ != NULL)
	{
		delete kdTree_;
		kdTree_ = NULL;
	}
}

void MSFiltering::denoise()
{
	MeshFilterParameters param;
	parameter_set_->getValue(QString("Max Iter."), param.max_iter);
	//parameter_set_->getValue(QString("lambda"), param.lambda);
	//parameter_set_->getValue(QString("eta"), param.eta);
	//parameter_set_->getValue(QString("mu"), param.mu);
	//parameter_set_->getValue(QString("nu"), param.nu);
	//if (!param.valid_parameters()) {
	//	std::cerr << "Invalid filter options. Aborting..." << std::endl;
	//	return;
	//}
	TriMesh mesh = data_manager_->getOriginalMesh();

	#ifdef USE_OPENMP
	Eigen::initParallel();
	#endif

	// Normalize the input mesh
	//Eigen::Vector3d original_center;
	//double original_scale;
	//normalize_mesh(mesh, original_center, original_scale);

	// update face normal
	Timer timer;
	Timer::EventID mesh_flter_begin_time = timer.get_time();
	std::vector<TriMesh::Normal> filtered_normal;
	updateFilteredNormals(mesh, filtered_normal);
	
	Timer::EventID update_begin_time = timer.get_time();
	target_normals_.resize(3, mesh_.n_faces());
	for (TriMesh::ConstFaceIter cf_it = mesh_.faces_begin(); cf_it != mesh_.faces_end(); ++cf_it)
	{
		Eigen::Vector3d f_normal = to_eigen_vec3d(filtered_normal.at(cf_it->idx()));
		target_normals_.col(cf_it->idx()) = f_normal;
	}

	TriMesh output_mesh;
	// update vertex
	std::cout << std::endl;
	std::cout << "====== Filter Finish =========" << std::endl;
	//param.output();
	iterative_mesh_update(param, target_normals_, output_mesh);

	//MeshNormalFilter mesh_filter(mesh);
	//mesh_filter.filter(param, output_mesh);
	//restore_mesh(output_mesh, original_center, original_scale);

	Timer::EventID update_end_time = timer.get_time();
	if (true) {
		std::cout << "Mesh udpate timing: " << timer.elapsed_time(update_begin_time, update_end_time) << " secs" << std::endl;
		std::cout << "Mesh filter total timing: " << timer.elapsed_time(mesh_flter_begin_time, update_end_time) << " secs" << std::endl;
	}

	data_manager_->setMesh(output_mesh);
	data_manager_->setDenoisedMesh(output_mesh);
}

void MSFiltering::initParameters()
{
	parameter_set_->removeAllParameter();

	parameter_set_->addParameter(QString("Iteration Num."), 0, QString("Iteration Num."), QString("The total iteration number of the algorithm."),
		true, 0, 100);
	parameter_set_->addParameter(QString("Multiple(* sigma_s)"), 3.0, QString("Multiple(* sigma_s)"), QString("Standard deviation of spatial weight."),
		true, 1.0e-9, 100.0);
	parameter_set_->addParameter(QString("Multiple(* sigma_r)"), 0.2, QString("Multiple(* sigma_r)"), QString("Standard deviation of range weight."),
		true, 1.0e-9, 100.0);

	parameter_set_->addParameter(QString("K"), 0.2, QString("K"), QString("Standard deviation of range weight."),
		true, 1.0e-9, 100.0);
	parameter_set_->addParameter(QString("Shrink Rate"), 1.0, QString("Shrink Rate"), QString("Standard deviation of range weight."),
		true, 1.0e-9, 100.0);
	parameter_set_->addParameter(QString("Neighbor Size"), 3, QString("Neighbor Size"), QString("The size of neighbor based on radius or ring"));
	parameter_set_->addParameter(QString("Response"), 20, QString("Response"), QString("The size of neighbor based on radius or ring"));
	parameter_set_->addParameter(QString("Max Iter."), 5, QString("Max Iter."), QString("The total iteration number of the algorithm."));


	parameter_set_->setName(QString("Rolling Guided Normal Filtering"));
	parameter_set_->setLabel(QString("Rolling Guided Normal Filtering"));
	parameter_set_->setIntroduction(QString("Rolling Guided Mesh Filtering -- Parameters"));
}

void MSFiltering::updateFilteredNormals(TriMesh & mesh, std::vector<TriMesh::Normal>& filtered_normals)
{

	filtered_normals = std::vector<TriMesh::Normal>(mesh.n_faces(), TriMesh::Normal(0.0, 0.0, 0.0));
	//get parameter
	int iteration_number;
	double sigma_s;
	double sigma_r;
	int neighbor_size_factor;
	double shrink_rate;

	if (!parameter_set_->getValue(QString("Iteration Num."), iteration_number)) return;
	if (!parameter_set_->getValue(QString("Multiple(* sigma_s)"), sigma_s)) return;
	if (!parameter_set_->getValue(QString("Multiple(* sigma_r)"), sigma_r)) return;
	if (!parameter_set_->getValue(QString("Neighbor Size"), neighbor_size_factor)) return;
	if (!parameter_set_->getValue(QString("Shrink Rate"), shrink_rate)) return;
	if (iteration_number == 0)
	{
		std::vector<TriMesh::Normal> all_face_normal;
		getFaceNormal(data_manager_->getOriginalMesh(), all_face_normal);
		filtered_normals = all_face_normal;
	}

	std::vector<double> all_face_area;
	getFaceArea(mesh, all_face_area);
	std::vector<TriMesh::Normal> all_face_normal;
	getFaceNormal(mesh, all_face_normal);

	
	std::vector<TriMesh::Point> face_centriod;
	getFaceCentroid(mesh, face_centriod);

	double mean_edge_length;
	getGlobalMeanEdgeLength(mesh, mean_edge_length);
	double radius = mean_edge_length;
	
	std::vector<double> values(mesh.n_faces());
	double original_sigma = sigma_s * mean_edge_length;
	std::vector<double> sig_s;// (mesh.n_faces(), original_sigma);
	std::vector<double> sig_r;// (mesh.n_faces(), sigma_r);
	
	std::vector<std::vector<double>> all_face_neighbor_dist;
	std::vector<std::vector<int>> all_face_neighbor_index;
	
	// get all face neighbor and sigma_s and sigma_r
	getAllFaceNeighbor(mesh, face_centriod, sig_s, sig_r, original_sigma, sigma_r, 
		all_face_neighbor_dist, all_face_neighbor_index);
	//initSigma(mesh, 0, sig_s, sig_r, original_sigma, sigma_r);
	Timer timer;
	Timer::EventID start_filter = timer.get_time();
	for (int k = 0; k < iteration_number; k++)
	{
		std::vector<TriMesh::Normal> temp_normals = std::vector<TriMesh::Normal>(mesh.n_faces(), TriMesh::Normal(0.0, 0.0, 0.0));

		double average_faces_size = 0.0;
		#pragma omp parallel
		{
			#pragma omp for reduction(+:average_faces_size)
			for (int index_i = 0; index_i < int(mesh.n_faces()); index_i++)
			{

				//double t = temparature.at(index_i);
				TriMesh::Normal ni_k = filtered_normals[index_i];
				TriMesh::Normal n_temp(0.0, 0.0, 0.0);
				TriMesh::Point ci = face_centriod[index_i];

				// get neighbor


				// init sigma_s sigma_r
				double curr_sigma_s = sig_s.at(index_i);
				double curr_sigma_r = sig_r.at(index_i);

				std::vector<int> idxs; // all_face_neighbor_idx[index_i];
				std::vector<double> dists; // all_face_neighbor_dist[index_i];
				idxs = all_face_neighbor_index.at(index_i);
				dists = all_face_neighbor_dist.at(index_i);

				//double sqRad = pow(curr_sigma_s*3, 2);
				//annSearchFaceNeighor(ci, sqRad, idxs, dists);

				// compute new tempurature
				double cov = getCovariance(idxs, dists, all_face_area, filtered_normals, curr_sigma_s / 2);
				sig_s.at(index_i) = sig_s.at(index_i) / ((1 + shrink_rate * cov));
				sig_r.at(index_i) = sig_r.at(index_i) / ((1 + shrink_rate * cov));
				// update curr_sigma_s
				curr_sigma_s = sig_s.at(index_i);
				curr_sigma_r = sig_r.at(index_i);

				int cnt = 0;
				// Compute Rolling input new sigma_s new sigma_r
				double weight_sum = 0.0;
				for (int j = 0; j < (int)idxs.size(); j++)
				{
					double spatial_distance = dists[j];
					if (spatial_distance > pow((3 * curr_sigma_s), 2)) break;
					cnt++;

					int index_j = idxs[j];
					TriMesh::Normal nj_k = filtered_normals[index_j];
					TriMesh::Normal nj = all_face_normal[index_j];
					TriMesh::Point cj = face_centriod[index_j];
					double spatial_weight = GaussianWeight(spatial_distance, curr_sigma_s);
					double range_distance = (ni_k - nj_k).length() * (ni_k - nj_k).length();
					double range_weight = GaussianWeight(range_distance, curr_sigma_r);
					double face_area = all_face_area[index_j];
					double weight = face_area * range_weight * spatial_weight;
					weight_sum += weight;
					n_temp += weight * nj;
				}
				// update temp_normals
				n_temp /= weight_sum;
				n_temp.normalize();
				temp_normals[index_i] = n_temp;
				values.at(index_i) = sig_s.at(index_i) / original_sigma;
				// compute total
				average_faces_size += cnt;
			}
		}
		//update filtered_normals for iteration
		std::cout << "Aver. neighbor size " << average_faces_size / mesh.n_faces() << std::endl;
		filtered_normals = temp_normals;
	}
	Timer::EventID end = timer.get_time();
	std::cout << "Iterative time total: " << timer.elapsed_time(start_filter, end) << " secs" << std::endl;

	double max_ele = *max_element(values.begin(), values.end());
	double min_ele = *min_element(values.begin(), values.end());
	std::cout << "Value max: " << max_ele << "Value Min: "<<min_ele << std::endl;

	data_manager_->setRenderValues(values);
	data_manager_->setFilteredNormal(filtered_normals);
}

void MSFiltering::getGlobalMeanEdgeLength(TriMesh & mesh, double & mean_edge_length)
{
	int num = 0;

	double sum_edge_length = 0.0;
	for (TriMesh::EdgeIter e_it = mesh.edges_begin(); e_it != mesh.edges_end(); e_it++)
	{
		TriMesh::EdgeHandle eh = *e_it;
		sum_edge_length += mesh.calc_edge_length(eh);
		num++;
	}
	mean_edge_length = sum_edge_length / num;
}

double MSFiltering::getCovariance(const std::vector<int>& idxs2, const std::vector<double>& dists, const std::vector<double>& all_face_area, const std::vector<TriMesh::Normal>& normals, double gaussian)
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
	mean_normal /= (normalizer+10e-8);

	double var = 0.0;
	for (int i = 0; i < int(idxs2.size()); i++)
	{
		TriMesh::Normal nr_jk = normals[idxs2[i]];
		double curr_area = all_face_area[idxs2[i]];
		double spatial_distance = dists[i];
		double g = GaussianWeight(spatial_distance, gaussian);
		var += g * curr_area * pow((mean_normal - nr_jk).length(), 2);
	}
	var /= (normalizer+10e-8);
	return var;
}



void MSFiltering::getLaplacianNormal(TriMesh & mesh, const std::vector<TriMesh::Normal>& all_face_normal, std::vector<TriMesh::Normal>& gradient_normal)
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


void MSFiltering::annSearchFaceNeighor(TriMesh::Point queryPt, double radius, std::vector<int>& idxs, std::vector<double>& dists)
{
	ANNdist dist = radius;
	int k_radius;
	//ANNdist radius = 1;
	// convert to ann point
	ANNpoint annPt = toAnnPt(queryPt);



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
	if (nnIdx != NULL)
		delete nnIdx;
	if (nndists != NULL)
		delete nndists;
}
