#include "SDMeshFiltering.h"


SDMeshDenoisying::SDMeshDenoisying(DataManager * _data_manager, ParameterSet * _parameter_set)
:MeshDenoisingBase(_data_manager, _parameter_set)
{
	initParameters();
}

SDMeshDenoisying::~SDMeshDenoisying()
{

}

void SDMeshDenoisying::denoise()
{
	SDFilter::MeshFilterParameters param;
	parameter_set_->getValue(QString("Max Iter."), param.max_iter);
	parameter_set_->getValue(QString("lambda"), param.lambda);
	parameter_set_->getValue(QString("eta"), param.eta);
	parameter_set_->getValue(QString("mu"), param.mu);
	parameter_set_->getValue(QString("nu"), param.nu);
	if (!param.valid_parameters()) {
		std::cerr << "Invalid filter options. Aborting..." << std::endl;
		return;
	}
	param.output();
	TriMesh mesh = data_manager_->getOriginalMesh();

	#ifdef USE_OPENMP
	Eigen::initParallel();
	#endif
	
	// Normalize the input mesh
	Eigen::Vector3d original_center;
	double original_scale;
	SDFilter::normalize_mesh(mesh, original_center, original_scale);
	
	TriMesh output_mesh;
	SDFilter::MeshNormalFilter mesh_filter(mesh);
	mesh_filter.filter(param, output_mesh);
	SDFilter::restore_mesh(output_mesh, original_center, original_scale);
	
	data_manager_->setMesh(output_mesh);
	data_manager_->setDenoisedMesh(output_mesh);
}

void SDMeshDenoisying::initParameters()
{
	parameter_set_->removeAllParameter();

	parameter_set_->addParameter(QString("Max Iter."), 5, QString("Max Iter."), QString("The total iteration number of the algorithm."));
	parameter_set_->addParameter(QString("lambda"), 10.0, QString("lambda"), QString("lambda"), true, 0.0, 100.0);
	parameter_set_->addParameter(QString("eta"), 1.0 , QString("eta"), QString("eta"), true, 0.0, 100.0);
	
	parameter_set_->addParameter(QString("mu"), 1.0, QString("mu"), QString("mu"), true, 0.0, 100.0);
	parameter_set_->addParameter(QString("nu"), 1.0, QString("nu"), QString("nu"), true, 0.0, 100.0);

	parameter_set_->setName(QString("SD Mesh Denoising"));
	parameter_set_->setLabel(QString("SD Mesh Denoising"));
	parameter_set_->setIntroduction(QString("SD Mesh Denoising -- Paramters"));
}
