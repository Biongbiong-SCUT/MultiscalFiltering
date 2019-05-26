#pragma once
#include "MSFiltering.h"
#include "Algorithms/MeshDenoisingBase.h"
#include "Algorithms/SDMeshFiltering.h"
#include "Algorithms\SDFilter\MeshNormalDenoising.h"
//include "Algorithms\SDFilter\SDFilter.h"
#include "Algorithms/SDFilter/SDFilter.h"
#include "ANN/ANN.h"
#include "Algorithms\SDFilter\EigenTypes.h"

#include <algorithm>

// Linear system solver
using namespace SDFilter;
class LinearSolver
{
public:
	LinearSolver(Parameters::LinearSolverType solver_type)
		:solver_type_(solver_type), pattern_analyzed(false) {}

	// Initialize the solver with matrix
	bool compute(const SparseMatrixXd &A)
	{
		if (solver_type_ == Parameters::LDLT)
		{
			if (!pattern_analyzed)
			{
				LDLT_solver_.analyzePattern(A);
				if (!check_error(LDLT_solver_, "Cholesky analyzePattern failed")) {
					return false;
				}

				pattern_analyzed = true;
			}

			LDLT_solver_.factorize(A);
			return check_error(LDLT_solver_, "Cholesky factorization failed");
		}
		else if (solver_type_ == Parameters::CG) {
			CG_solver_.compute(A);
			return check_error(CG_solver_, "CG solver compute failed");
		}
		else {
			return false;
		}
	}

	template<typename MatrixT>
	bool solve(const MatrixT &rhs, MatrixT &sol)
	{
		if (solver_type_ == Parameters::LDLT)
		{
#ifdef USE_OPENMP
			int n_cols = rhs.cols();

			OMP_PARALLEL
			{
				OMP_FOR
				for (int i = 0; i < n_cols; ++i) {
					sol.col(i) = LDLT_solver_.solve(rhs.col(i));
				}
			}
#else
			sol = LDLT_solver_.solve(rhs);
#endif

			return check_error(LDLT_solver_, "LDLT solve failed");
		}
		else if (solver_type_ == Parameters::CG)
		{
			sol = CG_solver_.solve(rhs);
			return check_error(CG_solver_, "CG solve failed");
		}
		else {
			return false;
		}
	}

	void reset_pattern()
	{
		pattern_analyzed = false;
	}

	void set_solver_type(Parameters::LinearSolverType type)
	{
		solver_type_ = type;
		if (solver_type_ == Parameters::LDLT) {
			reset_pattern();
		}
	}

private:
	Parameters::LinearSolverType solver_type_;
	Eigen::SimplicialLDLT<SparseMatrixXd> LDLT_solver_;
	Eigen::ConjugateGradient<SparseMatrixXd, Eigen::Lower | Eigen::Upper  > CG_solver_;

	bool pattern_analyzed;	// Flag for symbolic factorization

	template<typename SolverT>
	bool check_error(const SolverT &solver, const std::string &error_message) {
		if (solver.info() != Eigen::Success) {
			std::cerr << error_message << std::endl;
		}

		return solver.info() == Eigen::Success;
	}
};

class MSFiltering
	:public MeshDenoisingBase
{
public:
	MSFiltering(DataManager *_data_manager, ParameterSet *_parameter_set);
	~MSFiltering();
	void denoise();
	void initParameters();

	// Methods for mesh vertex update according to filtered normals
	enum MeshUpdateMethod
	{
		ITERATIVE_UPDATE,	// ShapeUp styled iterative solver
		POISSON_UPDATE,	// Poisson-based update from [Want et al. 2015]
	};

	void init_signals(Eigen::MatrixXd &init_signals)
	{
		init_signals.resize(3, mesh_.n_faces());

		for (TriMesh::ConstFaceIter cf_it = mesh_.faces_begin(); cf_it != mesh_.faces_end(); ++cf_it)
		{
			Eigen::Vector3d f_normal = to_eigen_vec3d(mesh_.calc_face_normal(*cf_it)).normalized();
			init_signals.col(cf_it->idx()) = f_normal;
		}
	}

	double GaussianWeight(double Sqdistance, double sigma)
	{
		return std::exp(-0.5 * Sqdistance / (sigma * sigma));
	}
	void updateFilteredNormals(TriMesh &mesh, std::vector<TriMesh::Normal> &filtered_normals);
	void getGlobalMeanEdgeLength(TriMesh &mesh, double& mean_edge_length);
	// covariance 
	double getCovariance(const std::vector<int>& idxs2, const std::vector<double>& dists, const std::vector<double>& all_face_area, const std::vector<TriMesh::Normal> &normals, double gaussian);
	void initSigma(TriMesh &mesh, double radius, std::vector<double> &sigma_s, std::vector<double> &sigma_r, double original_sigmas, double original_sigmar)
	{

		Timer timer;
		Timer::EventID start_init = timer.get_time();
		std::vector<TriMesh::Normal> g_normal;
		std::vector<TriMesh::Normal> all_face_normals;
		std::vector<TriMesh::Point> all_face_centorids;
		getFaceNormal(mesh, all_face_normals);
		getFaceCentroid(mesh, all_face_centorids);
		getLaplacianNormal(mesh, all_face_normals, g_normal);
		sigma_s.resize(mesh.n_faces());
		sigma_r.resize(mesh.n_faces());

		double min = +10e8;
		double max = -10e8;
		double Ave_size = 0.0;
		radius = pow(original_sigmas, 2);

		std::cout << "procs num " << omp_get_num_procs() << std::endl;; //获取执行核的数量
		std::cout << "max threads " << omp_get_max_threads() << std::endl;; //获取执行核的数量
		

		Timer::EventID start_RTV = timer.get_time();
		//#pragma omp parallel for reduction(+:Ave_size)
		for (int index = 0; index < int(all_face_centorids.size()); index++)
		{
			//TriMesh::Normal rtv = TriMesh::Normal(0.0, 0.0, 0.0);
			TriMesh::Point ci = all_face_centorids.at(index);
			std::vector<int> idxs; // all_face_neighbor_idx[index_i];
			std::vector<double> dists; // all_face_neighbor_dist[index_i];
			//#pragma omp critical
			{
				annSearchFaceNeighor(ci, radius, idxs, dists);
			}

			double x = 0.0, y = 0.0, z = 0.0;
			double normalizer = 0.0;
			for (int i = 0; i < int(idxs.size()); i++)
			{
				double spatial_distance = dists[i];
				double g = GaussianWeight(spatial_distance, radius);
				TriMesh::Normal no_jk = g_normal[idxs[i]];// -no_i;
				x += g * pow(abs(no_jk[0]), 2);
				y += g * pow(abs(no_jk[1]), 2);
				z += g * pow(abs(no_jk[2]), 2);
				normalizer += g;
			}

			//pragma omp critical
			{
				if (min > (x + y + z) / normalizer)
				{
					min = (x + y + z) / normalizer;
				}
				if (max < (x + y + z) / normalizer)
				{
					max = (x + y + z) / normalizer;
				}
			}
			sigma_s.at(index) = (x + y + z) / normalizer;
			sigma_r.at(index) = (x + y + z) / normalizer;
			Ave_size += idxs.size();
		}

		int coeff = 0;
		double k = 0.0;
		parameter_set_->getValue("K", k);
		parameter_set_->getValue("Response", coeff);
		std::cout << "response " << coeff << std::endl;
		std::cout << "K " << k << std::endl;
		std::cout << "Region Diff max: " << max << "Regin Diff min: " << min << std::endl;

		Timer::EventID start = timer.get_time();

		OMP_PARALLEL
		{
			OMP_FOR
			for (int i = 0; i < int(sigma_s.size()); ++i)
			{
				double response = coeff * sigma_s.at(i);
				double value = 2 / (1 + exp(-response)) - 1;
				sigma_s.at(i) = original_sigmas * (k * value + (1 - k));
				sigma_r.at(i) = original_sigmar * (k * value + (1 - k));
			}
		}
		
		Timer::EventID end = timer.get_time();

		//Timer::EventID endend = timer.get_time();
		//std::cout << "USE OMP: " << timer.elapsed_time(start, end) << " secs" << std::endl;
		std::cout << "init Total: " << timer.elapsed_time(start_init, end) << " secs" << std::endl;
		std::cout << "RTV total: " << timer.elapsed_time(start_RTV, start) << " secs" << std::endl;
		
		double max_ele = *max_element(sigma_s.begin(), sigma_s.end());
		double min_ele = *min_element(sigma_s.begin(), sigma_s.end());
		std::cout << "Response Max: " << max_ele / original_sigmas << "Response Min: " << min_ele << std::endl;
		std::cout << "init Sigma Ave.Neighbor Size " << Ave_size / mesh.n_faces() << std::endl;
	}

	void getLaplacianNormal(TriMesh &mesh, const std::vector<TriMesh::Normal> &all_face_normal, std::vector<TriMesh::Normal> &gradient_normal);
	
	void getAllFaceNeighbor(TriMesh &mesh, std::vector<TriMesh::Point> &all_face_centriod, std::vector<double> &sigma_s, std::vector<double> &sigma_r, double original_sigmas, double original_sigmar,
		std::vector<std::vector<double>> &all_face_neighbor_dist, std::vector<std::vector<int>> &all_face_neighbor_index)
	{
		initSigma(mesh, 0, sigma_s, sigma_r, original_sigmas, original_sigmar);
		all_face_neighbor_dist.clear();
		all_face_neighbor_index.clear();
		all_face_neighbor_dist.resize(mesh.n_faces());
		all_face_neighbor_index.resize(mesh.n_faces());
		Timer timer;
		Timer::EventID get_negibor = timer.get_time();
		int size = 0;
		//for (std::vector<TriMesh::Point>::iterator c_it = all_face_centriod.begin(); c_it!=all_face_centriod.end(); ++c_it)
		OMP_PARALLEL
		{
			OMP_FOR
			for (int i = 0; i < all_face_centriod.size(); i++)
			{
				std::vector<int> idxs; // all_face_neighbor_idx[index_i];
				std::vector<double> dists; // all_face_neighbor_dist[index_i];
				double curr_sigma_s = sigma_s[i];
				double sqRad = pow(curr_sigma_s * 3, 2);
				#pragma omp critical 
				{
					annSearchFaceNeighor(all_face_centriod[i], sqRad, idxs, dists);

				}
				all_face_neighbor_dist.at(i) = dists;
				all_face_neighbor_index.at(i) = idxs;
				size += idxs.size();
			}
		}
		Timer::EventID end = timer.get_time();
		
		std::cout << "All Face Neighbor Occpy Dist " << size * sizeof(double) / (1024 * 1024.0) << "M" << std::endl;
		std::cout << "All Face Neighbor Occpy Index " << size * sizeof(int) / (1024 * 1024.0) << "M" << std::endl;
		std::cout << "Neighbor total: " << timer.elapsed_time(get_negibor, end) << " secs" << std::endl;
	}

	void getWeight(TriMesh &mesh, const std::vector<TriMesh::Normal> &u_normals, const std::vector<TriMesh::Normal> &g_normals, const std::vector<double>& all_face_area, std::vector<int>idxs, std::vector<double>dists, double guassian)
	{
		
		double cov = getCovariance(idxs, dists, all_face_area, u_normals, guassian);
		// compute rtv
		TriMesh::Normal rtv = TriMesh::Normal(0.0, 0.0, 0.0);
		double area_sum = 0.0;
		double x = 0.0, y = 0.0, z = 0.0;
		for (int i = 0; i < int(idxs.size()); i++)
		{
			double spatial_distance = dists[i];
			double g = GaussianWeight(spatial_distance, guassian);
			TriMesh::Normal no_jk = g_normals[idxs[i]];// -no_i;
			rtv += g * no_jk;
			x += g * pow(abs(no_jk[0]), 2);
			y += g * pow(abs(no_jk[1]), 2);
			z += g * pow(abs(no_jk[2]), 2);
		}

		double region_rtv = x + y + z;
		
		double response = 20 * (cov - region_rtv);
		double weights = 1 / (1 + exp(-response));
	}
protected:
	TriMesh mesh_;

	Eigen::MatrixXd signals_;
	Matrix3X target_normals_;
	
	
	// for neighborhood search
	inline ANNpoint toAnnPt(TriMesh::Point pt) {
		ANNpoint annPt = annAllocPt(3);
		annPt[0] = pt[0];
		annPt[1] = pt[1];
		annPt[2] = pt[2];
		return annPt;;
	}

	void initKDtree()
	{
		std::cout << "Start Building KD Tree for KNN search" << std::endl;
		std::vector<TriMesh::Point> all_face_centroid;
		TriMesh mesh = data_manager_->getNoisyMesh();
		getFaceCentroid(mesh, all_face_centroid);
		
		int nPts = all_face_centroid.size();			//顶点数
		dataPts_ = annAllocPts(nPts, 3);
	
		for (TriMesh::ConstFaceIter cf_it = mesh_.faces_begin(); cf_it != mesh_.faces_end(); ++cf_it)
		{
			dataPts_[cf_it->idx()] = toAnnPt(all_face_centroid[cf_it->idx()]);
		}

		kdTree_ = new ANNkd_tree(dataPts_, nPts, 3);

		std::cout << "====== Build KD Tree Finish =========" << std::endl;
	}
	void annSearchFaceNeighor(TriMesh::Point queryPt, double radius, std::vector<int>& idxs, std::vector<double>& dists);
	
	
	ANNkd_tree*	kdTree_;
	ANNpointArray	dataPts_;

private:

	// solver
	// Linear solver for symmetric positive definite matrix,
	LinearSolver linear_solver_;	// Linear system solver for mesh update
	SparseMatrixXd At_;		// Transpose of part of the linear least squares matrix that corresponds to mean centering of face vertices
	bool system_matrix_factorized_;	// Whether the matrix


	// Set up and pre-factorize the linear system for iterative mesh update
	bool setup_mesh_udpate_system(const Matrix3Xi &face_vtx_idx, double w_closeness)
	{
		if (system_matrix_factorized_)
		{
			return true;
		}

		int n_faces = mesh_.n_faces();
		int n_vtx = mesh_.n_vertices();
		std::vector<Triplet> A_triplets(9 * n_faces);
		std::vector<Triplet> I_triplets(n_vtx);

		// Matrix for mean centering of three vertices
		Eigen::Matrix3d mean_centering_mat;
		get_mean_centering_matrix(mean_centering_mat);

		OMP_PARALLEL
		{
			OMP_FOR
			for (int i = 0; i < n_faces; ++i)
			{
				Eigen::Vector3i vtx_idx = face_vtx_idx.col(i);

				int triplet_addr = 9 * i;
				int row_idx = 3 * i;
				for (int j = 0; j < 3; ++j)
				{
					for (int k = 0; k < 3; ++k) {
						A_triplets[triplet_addr++] = Triplet(row_idx, vtx_idx(k), mean_centering_mat(j, k));
					}

					row_idx++;
				}
			}

			OMP_FOR
			for (int i = 0; i < n_vtx; ++i)
			{
				I_triplets[i] = Triplet(i, i, w_closeness);
			}
		}


		SparseMatrixXd A(3 * n_faces, n_vtx);
		A.setFromTriplets(A_triplets.begin(), A_triplets.end());
		At_ = A.transpose();
		At_.makeCompressed();

		SparseMatrixXd wI(n_vtx, n_vtx);
		wI.setFromTriplets(I_triplets.begin(), I_triplets.end());
		SparseMatrixXd M = At_ * A + wI;

		linear_solver_.reset_pattern();
		if (!linear_solver_.compute(M)) {
			std::cerr << "Error: failed to pre-factorize mesh update system" << std::endl;
			return false;
		}

		system_matrix_factorized_ = true;
		return true;
	}


	void get_face_area_weights(const TriMesh &mesh, Eigen::VectorXd &face_area_weights) const
	{
		face_area_weights.resize(mesh.n_faces());

		for (TriMesh::ConstFaceIter cf_it = mesh.faces_begin(); cf_it != mesh.faces_end(); ++cf_it)
		{
			face_area_weights(cf_it->idx()) = mesh.calc_sector_area(mesh.halfedge_handle(*cf_it));
		}

		face_area_weights /= face_area_weights.mean();
	}


	bool iterative_mesh_update(const MeshFilterParameters &param, const Matrix3X &target_normals, TriMesh &output_mesh)
	{
		// Rescale closeness weight using the ratio between face number and vertex number, and take its square root
		double w_closeness = param.mesh_update_closeness_weight * double(mesh_.n_faces()) / mesh_.n_vertices();

		output_mesh = mesh_;

		Matrix3Xi face_vtx_idx;
		get_face_vertex_indices(output_mesh, face_vtx_idx);

		if (!setup_mesh_udpate_system(face_vtx_idx, w_closeness)) {
			return false;
		}

		std::cout << "Starting iterative mesh update......" << std::endl;

		Matrix3X vtx_pos;
		get_vertex_points(output_mesh, vtx_pos);

		int n_faces = output_mesh.n_faces();
		Eigen::Matrix3Xd target_plane_local_frames(3, 2 * n_faces);	// Local frame for the target plane of each face
		std::vector<bool> local_frame_initialized(n_faces, false);

		Eigen::MatrixX3d wX0 = vtx_pos.transpose() * w_closeness;	// Part of the linear system right-hand-side that corresponds to initial vertex positions
		Eigen::MatrixX3d B(3 * n_faces, 3);	// Per-face target position of the new vertices

		int n_vtx = output_mesh.n_vertices();
		Eigen::MatrixX3d rhs(n_vtx, 3), sol(n_vtx, 3);

		for (int iter = 0; iter < param.mesh_update_iter; ++iter)
		{
			OMP_PARALLEL
			{
				OMP_FOR
				for (int i = 0; i < n_faces; ++i)
				{
					Eigen::Vector3d current_normal = to_eigen_vec3d(output_mesh.calc_face_normal(TriMesh::FaceHandle(i)));
					Eigen::Vector3d target_normal = target_normals.col(i);

					Eigen::Matrix3d face_vtx_pos;
					get_mean_centered_face_vtx_pos(vtx_pos, face_vtx_idx.col(i), face_vtx_pos);

					Eigen::Matrix3Xd target_pos;

					// If the current normal is not pointing away from the target normal, simply project the points onto the target plane
					if (current_normal.dot(target_normal) >= 0) {
						target_pos = face_vtx_pos - target_normal * (target_normal.transpose() * face_vtx_pos);
					}
					else {
						// Otherwise, project the points onto a line in the target plane
						typedef Eigen::Matrix<double, 3, 2> Matrix32d;
						Matrix32d current_local_frame;
						if (local_frame_initialized[i]) {
							current_local_frame = target_plane_local_frames.block(0, 2 * i, 3, 2);
						}
						else {
							Eigen::JacobiSVD<Eigen::Vector3d, Eigen::FullPivHouseholderQRPreconditioner> jSVD_normal(target_normal, Eigen::ComputeFullU);
							current_local_frame = jSVD_normal.matrixU().block(0, 1, 3, 2);
							target_plane_local_frames.block(0, 2 * i, 3, 2) = current_local_frame;
							local_frame_initialized[i] = true;
						}

						Matrix32d local_coord = face_vtx_pos.transpose() * current_local_frame;
						Eigen::JacobiSVD<Matrix32d> jSVD_coord(local_coord, Eigen::ComputeFullV);
						Eigen::Vector2d fitting_line_direction = jSVD_coord.matrixV().col(0);
						Eigen::Vector3d line_direction_3d = current_local_frame * fitting_line_direction;
						target_pos = line_direction_3d * (line_direction_3d.transpose() * face_vtx_pos);
					}

					B.block(3 * i, 0, 3, 3) = target_pos.transpose();
				}
			}

				// Solver linear system
			rhs = At_ * B + wX0;
			if (!linear_solver_.solve(rhs, sol)) {
				std::cerr << "Error: failed to solve mesh update system" << std::endl;
				return false;
			}

			vtx_pos = sol.transpose();
			set_vertex_points(output_mesh, vtx_pos);
		}

		return true;
	}

	// Generate the matrix for mean-centering of the vertices of a triangle
	void get_mean_centering_matrix(Eigen::Matrix3d &mat)
	{
		mat = Eigen::Matrix3d::Identity() - Eigen::Matrix3d::Constant(1.0 / 3);
	}

	void get_mean_centered_face_vtx_pos(const Eigen::Matrix3Xd &vtx_pos, const Eigen::Vector3i &face_vtx, Eigen::Matrix3d &face_vtx_pos)
	{
		for (int i = 0; i < 3; ++i) {
			face_vtx_pos.col(i) = vtx_pos.col(face_vtx(i));
		}

		Eigen::Vector3d mean_pt = face_vtx_pos.rowwise().mean();
		face_vtx_pos.colwise() -= mean_pt;
	}

	// Compute the centroid of a mesh given its vertex positions and face areas
	Eigen::Vector3d compute_centroid(const Eigen::Matrix3Xi &face_vtx_idx, const Eigen::VectorXd &face_areas, const Eigen::Matrix3Xd &vtx_pos)
	{
		int n_faces = face_vtx_idx.cols();
		Eigen::Matrix3Xd face_centroids(3, n_faces);

		OMP_PARALLEL
		{
			OMP_FOR
			for (int i = 0; i < n_faces; ++i)
			{
				Eigen::Vector3d c = Eigen::Vector3d::Zero();
				Eigen::Vector3i face_vtx = face_vtx_idx.col(i);

				for (int j = 0; j < 3; ++j) {
					c += vtx_pos.col(face_vtx(j));
				}

				face_centroids.col(i) = c / 3.0;
			}
		}

		return (face_centroids * face_areas) / face_areas.sum();
	}


	////////// Methods for evaluating the quality of the updated mesh /////////////

	// Compute the L2 norm between the initial mesh and filtered mesh
	void show_normalized_mesh_displacement_norm(const TriMesh &filtered_mesh)
	{
		Eigen::Matrix3Xd init_vtx_pos, new_vtx_pos;
		get_vertex_points(mesh_, init_vtx_pos);
		get_vertex_points(filtered_mesh, new_vtx_pos);
		Eigen::VectorXd vtx_disp_sqr_norm = (init_vtx_pos - new_vtx_pos).colwise().squaredNorm();

		// Computer normalized vertex area weights from the original mesh
		Eigen::VectorXd face_area_weights;
		get_face_area_weights(mesh_, face_area_weights);
		Eigen::Matrix3Xi face_vtx_indices;
		get_face_vertex_indices(mesh_, face_vtx_indices);
		int n_faces = mesh_.n_faces();

		Eigen::VectorXd vtx_area(mesh_.n_vertices());
		vtx_area.setZero();
		for (int i = 0; i < n_faces; ++i) {
			for (int j = 0; j < 3; ++j) {
				vtx_area(face_vtx_indices(j, i)) += face_area_weights(i);
			}
		}
		vtx_area /= vtx_area.sum();

		std::cout << "Normalized mesh displacement norm: " <<
			std::sqrt(vtx_area.dot(vtx_disp_sqr_norm)) / average_edge_length(mesh_) << std::endl;
	}


	void show_error_statistics(const Eigen::VectorXd &err_values, double bin_size, int n_bins)
	{
		int n_elems = err_values.size();


		

		Eigen::VectorXi error_bin_idx(n_elems);
		OMP_PARALLEL
		{
			OMP_FOR
			for (int i = 0; i < n_elems; ++i) {
				error_bin_idx(i) = std::min(n_bins, static_cast<int>(std::floor(err_values(i) / bin_size)));
			}
		}

		Eigen::VectorXd bin_count(n_bins + 1);
		bin_count.setZero();

		for (int i = 0; i < n_elems; ++i)
		{
			bin_count(error_bin_idx(i)) += 1;
		}

		bin_count /= bin_count.sum();

		for (int i = 0; i < n_bins; ++i)
		{
			double lower_val = bin_size * i;
			double upper_val = bin_size * (i + 1);
			std::cout << lower_val << " to " << upper_val << ": " << bin_count(i) * 100 << "%" << std::endl;
		}

		std::cout << "Over " << bin_size * n_bins << ": " << bin_count(n_bins) * 100 << "%" << std::endl;
	}

	// Show statistics of the deviation between the new normals and target normals (in degrees)
	void show_normal_error_statistics(const TriMesh &mesh, const Matrix3X &target_normals, int bin_size_in_degrees, int n_bins)
	{
		// Compute the normal deviation angle, and the number of flipped normals
		int n_faces = mesh.n_faces();
		Eigen::VectorXd face_normal_error_angle(n_faces);

		for (int i = 0; i < n_faces; ++i)
		{
			Eigen::Vector3d normal = to_eigen_vec3d(mesh.calc_face_normal(TriMesh::FaceHandle(i)));
			double error_angle_cos = std::max(-1.0, std::min(1.0, normal.dot(target_normals.col(i))));
			face_normal_error_angle(i) = std::acos(error_angle_cos);
		}

		face_normal_error_angle *= (180 / M_PI);

		std::cout << "Statistics of deviation between new normals and target normals:" << std::endl;
		std::cout << "===============================================================" << std::endl;
		show_error_statistics(face_normal_error_angle, bin_size_in_degrees, n_bins);
	}
	
};

