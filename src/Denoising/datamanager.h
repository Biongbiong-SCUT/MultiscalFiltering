#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include "mesh.h"
#include <string>

class DataManager
{
public:
    DataManager();

public:
    bool ImportMeshFromFile(std::string filename);
    bool ExportMeshToFile(std::string filename);
	bool ImportMeshFromFileToOriginal(std::string filename);

    TriMesh getMesh() const {return mesh_;}
    TriMesh getNoisyMesh() const {return noisy_mesh_;}
    TriMesh getOriginalMesh() const {return original_mesh_;}
    TriMesh getDenoisedMesh() const {return denoised_mesh_;}
    void getFilteredNormals(std::vector<TriMesh::Normal> &normals) const {normals = filtered_normals_;}
    void setMesh(const TriMesh &_mesh) {mesh_ = _mesh;}
    void setNoisyMesh(const TriMesh &_noisy_mesh) {noisy_mesh_ = _noisy_mesh;}
    void setOriginalMesh(const TriMesh &_original_mesh) {original_mesh_ = _original_mesh;}
    void setDenoisedMesh(const TriMesh &_denoised_mesh) {denoised_mesh_ = _denoised_mesh;}

    //add
    void setFilteredNormal(std::vector<TriMesh::Normal> &_filtered_normals) {
        filtered_normals_ = _filtered_normals;
    }
	void setRenderValues(std::vector<double> &values)
	{
		values_ = values;
	}
	void getRenderValues(std::vector<double> &_values) const
	{
		_values = values_;
	}

    void MeshToNoisyMesh() {mesh_ = noisy_mesh_;}
    void MeshToOriginalMesh() {mesh_ = original_mesh_;}
    void MeshToDenoisedMesh() {mesh_ = denoised_mesh_;}
    void ClearMesh() {
        TriMesh new_mesh;
        setMesh(new_mesh);
        setOriginalMesh(new_mesh);
        setNoisyMesh(new_mesh);
        setDenoisedMesh(new_mesh);
    }

private:
    TriMesh original_mesh_;
    TriMesh noisy_mesh_;
    TriMesh denoised_mesh_;
    TriMesh mesh_;
    //add
    std::vector<TriMesh::Normal> filtered_normals_;
	std::vector<double> values_;
};

#endif // DATAMANAGER_H
