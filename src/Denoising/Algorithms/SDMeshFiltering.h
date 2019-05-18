#pragma once
#include "Algorithms\MeshDenoisingBase.h"
#include "Algorithms\SDMeshFiltering.h"
#include "Algorithms\SDFilter\MeshNormalDenoising.h"
#include "Algorithms\SDFilter\SDFilter.h"


class SDMeshDenoisying :
	public MeshDenoisingBase
{
public:
	SDMeshDenoisying(DataManager *_data_manager, ParameterSet *_parameter_set);
	~SDMeshDenoisying();
	void denoise();
	void initParameters();
private:
	//SDFilter::Parameters param_;
};

