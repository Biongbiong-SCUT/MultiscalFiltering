#include "meshexaminer.h"
#include "glwrapper.h"
#include <QFileDialog>
#include <QString>

MeshExaminer::MeshExaminer()
    :draw_points_status_(false), draw_edges_status_(false), draw_faces_status_(true), draw_normals_status_(false), draw_filtered_normals_(false)
{

}

MeshExaminer::~MeshExaminer()
{

}

void MeshExaminer::resetMesh(const TriMesh &_mesh, bool _need_normalize)
{    
    updateMesh(_mesh);

    if(_need_normalize){
        TriMesh::Point max_coord, min_coord;
        if(meshBoundingBox(min_coord, max_coord)){
            setScene( (OpenMesh::Vec3f)((min_coord + max_coord)*0.5), 0.5*(max_coord - min_coord).norm());
        }
        else{
            setScene( OpenMesh::Vec3f(0, 0, 0), 1);
        }
    }
}

void MeshExaminer::updateMesh(const TriMesh &_mesh)
{
    mesh_show_ = _mesh;
    mesh_show_.request_face_normals();
    mesh_show_.request_vertex_normals();
    mesh_show_.update_normals();
}

void MeshExaminer::draw()
{
    if (draw_points_status_)
    {
        glBegin(GL_POINTS);
        for (TriMesh::VertexIter v_it = mesh_show_.vertices_begin();
             v_it != mesh_show_.vertices_end(); v_it++)
        {
            TriMesh::Normal normal = mesh_show_.normal(*v_it);
            TriMesh::Point point = mesh_show_.point(*v_it);
            glColor3f(0.0f,0.0f,1.0f);
            glNormal3f(normal[0], normal[1], normal[2]);
            glVertex3f(point[0], point[1], point[2]);
        }
        glEnd();
    }

    if (draw_faces_status_)
    {
        glBegin(GL_TRIANGLES);
        for (TriMesh::FaceIter f_it = mesh_show_.faces_begin();
             f_it != mesh_show_.faces_end(); f_it++)
        {
            TriMesh::Normal normal = mesh_show_.normal(*f_it);

            for(TriMesh::FaceHalfedgeIter fh_it = mesh_show_.fh_iter(*f_it);
                fh_it.is_valid(); fh_it++)
            {
                TriMesh::VertexHandle toV = mesh_show_.to_vertex_handle(*fh_it);
                TriMesh::Point point = mesh_show_.point(toV);
                glColor3f(0.7f,0.7f,0.0f);
                //if (!normals_show_.empty() )
                //{
                  //  float length = (normals_show_[fh_it->idx()] - normal).length();
                   // length /= 20.0;
                    //glColor3f(7.0f + length, 7.0f - length / 2, 0.0f);
                //}
                //glColor3f(7.0f, 0.7f, 0.0f);

                glNormal3f(normal[0], normal[1], normal[2]);
                glVertex3f(point[0], point[1], point[2]);
            }
        }
        glEnd();
    }

    if (draw_edges_status_)
    {
        for(TriMesh::FaceIter f_it = mesh_show_.faces_begin();
            f_it != mesh_show_.faces_end(); f_it++)
        {
            glBegin(GL_LINE_LOOP);
            for(TriMesh::FaceHalfedgeIter fh_it = mesh_show_.fh_iter(*f_it);
                fh_it.is_valid(); fh_it++)
            {
                TriMesh::VertexHandle toV = mesh_show_.to_vertex_handle(*fh_it);
                TriMesh::Normal normal = mesh_show_.normal(toV);
                TriMesh::Point point = mesh_show_.point(toV);
                glColor3f(0.3f, 0.3f, 0.3f);
                glNormal3f(normal[0], normal[1], normal[2]);
                glVertex3f(point[0], point[1], point[2]);
            }
            glEnd();
        }
    }

    //if (draw_normals_status)
    if (draw_normals_status_)
    {
        double length = mesh_show_.calc_edge_length(*mesh_show_.edges_begin()) * 3;
        glBegin(GL_LINES);
        for(TriMesh::FaceIter f_it = mesh_show_.faces_begin(); f_it != mesh_show_.faces_end(); f_it++)
        {
            TriMesh::Point p1 = mesh_show_.calc_face_centroid(*f_it);
            TriMesh::Normal normal = mesh_show_.calc_face_normal(*f_it);
            glColor3f(1.0f, 1.0f, 1.0f);
            glVertex3d(p1[0], p1[1], p1[2]);
            glVertex3f(p1[0] + normal[0] * length, p1[1] + normal[1] * length, p1[2] + normal[2] * length);
        }
        glEnd();
    }

    if (draw_filtered_normals_ && !normals_show_.empty())
    {
        double length = mesh_show_.calc_edge_length(*mesh_show_.edges_begin()) * 2;
        glBegin(GL_LINES);
        for(TriMesh::FaceIter f_it = mesh_show_.faces_begin(); f_it != mesh_show_.faces_end(); f_it++)
        {
            TriMesh::Point p1 = mesh_show_.calc_face_centroid(*f_it);
            TriMesh::Normal normal = normals_show_[f_it->idx()];
            glColor3f(0.3f, 0.3f, 0.3f);
            glVertex3d(p1[0], p1[1], p1[2]);
            glVertex3f(p1[0] + normal[0] * length, p1[1] + normal[1] * length, p1[2] + normal[2] * length);
            //TriMesh::Point end1 = p1 + normal * length * 0.4;
            //glDrawCylinder(p1, end1, GLColor(0.3, 0.3, 0.3), 0.003, false);
            //TriMesh::Point end2 = end1 + normal * length * 0.2;
            //glDrawCylinder(end1, end2, GLColor(0.3, 0.3, 0.3), 0.006, true);
        }
        glEnd();
    }

    //if (!normals_show_.empty())
    //{
    //    for (TriMesh::FaceIter f_it = mesh_show_.faces_begin();
    //         f_it != mesh_show_.faces_end(); f_it++)
    //    {
    //        TriMesh::Normal normal = mesh_show_.normal(*f_it);
    //        float length = (normal - normals_show_[f_it -> idx()]).length();
    //        length /= 10.0;
    //        //qDebug() << "length is " << length;
    //        glBegin(GL_TRIANGLES);
    //        //if(length > 0.3)
    //        //    glColor3f(1.0f, 0.7f, 0.0f);
    //        //else glColor3f(0.7f, 0.7f, 0.0f);
    //        glColor3f(0.7f + length * 3, 0.7f - length * 3, 0.0f);
    //        for(TriMesh::FaceHalfedgeIter fh_it = mesh_show_.fh_iter(*f_it);
    //            fh_it.is_valid(); fh_it++)
    //        {
    //            TriMesh::VertexHandle toV = mesh_show_.to_vertex_handle(*fh_it);
    //            TriMesh::Point point = mesh_show_.point(toV);
    //            glNormal3f(normal[0], normal[1], normal[2]);
    //            glVertex3f(point[0], point[1], point[2]);
    //        }
    //        glEnd();
    //    }
    //}

	if (!values_.empty()) {
		int k = 0;

		for (TriMesh::FaceIter f_it = mesh_show_.faces_begin();
			f_it != mesh_show_.faces_end(); f_it++) {
			double curr_value = values_[f_it->idx()];
			glBegin(GL_TRIANGLES);
			double RGB[3];
			glHSV2RGB(2.0 / 3.0 * (1- curr_value), 0.7981f, 0.8353f, RGB);
			//if (k == 0) {
			//	qDebug() << "RGB: " << "(" << RGB[0] << ", " << RGB[1] << ", " << RGB[2] << ")";
			//	k = 1;
			//}
			//glColor3f(0.5f + 0.5f * curr_value, 0.0f, 1.0f - 0.5f * curr_value);
			glColor3f(RGB[0], RGB[1], RGB[2]);
			//glColor3f(0.0f, 0.0f, 1.0f);
			TriMesh::Normal normal = mesh_show_.normal(*f_it);

			for (TriMesh::FaceHalfedgeIter fh_it = mesh_show_.fh_iter(*f_it);
				fh_it.is_valid(); fh_it++)
			{
				TriMesh::VertexHandle toV = mesh_show_.to_vertex_handle(*fh_it);
				TriMesh::Point point = mesh_show_.point(toV);
				glNormal3f(normal[0], normal[1], normal[2]);
				glVertex3f(point[0], point[1], point[2]);
			}
			glEnd();
		}
	}
}

bool MeshExaminer::meshBoundingBox(TriMesh::Point &min_coord, TriMesh::Point &max_coord)
{
    if(mesh_show_.n_vertices() == 0){
        return false;
    }

    TriMesh::ConstVertexIter cv_it = mesh_show_.vertices_begin(), cv_end(mesh_show_.vertices_end());
    min_coord = mesh_show_.point(*cv_it);
    max_coord = min_coord;

    for( ++ cv_it ; cv_it != cv_end; ++ cv_it){
        min_coord.minimize(mesh_show_.point(*cv_it));
        max_coord.maximize(mesh_show_.point(*cv_it));
    }

    return true;
}
