#ifndef MESHEXAMINER_H
#define MESHEXAMINER_H

#include "glexaminer.h"
#include "mesh.h"
//#include <QtOpenGL/QGLWidget>
#include "glwrapper.h"
class GLColor
{
public:
    GLColor(const float& _r = 0, const float& _g = 0, const float& _b = 0, const float& _a = 1.0):r(_r),g(_g),b(_b),a(_a){}
    GLColor(const QColor & qcolor)
    {
        int _r;
        int _g;
        int _b;
        qcolor.getRgb(&_r, &_g, &_b);
        r = _r / 255.0;
        g = _g / 255.0;
        b = _b / 255.0;
    }
    float r;
    float g;
    float b;
    float a;
};

static GLColor cBrown(0.7, 0, 0);
static GLColor cRed(1, 0, 0);
static GLColor cGreen(0, 1, 0);
static GLColor cBlue(0, 0, 1);
static GLColor cWhite(1, 1, 1);
static GLColor cBlack(0, 0, 0);
static GLColor cYellow(0.2, 1, 0.2);
static GLColor cOrange(0.6, 0.4, 0);
static GLColor cSnow(0.5, 0.5, 0.5);
static GLColor cPink(0.8, 0.1, 0.5);

class MeshExaminer : public GLExaminer
{
public:
    MeshExaminer();
    virtual ~MeshExaminer();

    // draw the scene
    virtual void draw();

    void resetMesh(const TriMesh &_mesh, bool _need_normalize);

    void updateMesh(const TriMesh &_mesh);

    void setDrawPointsStatus(bool _val){
        draw_points_status_ = _val;
    }

    void setDrawFacesStatus(bool _val){
        draw_faces_status_ = _val;
    }

    void setDrawEdgesStatus(bool _val){
        draw_edges_status_ = _val;
    }

    void setDrawNormaslStatus(bool _val) {
        draw_normals_status_ = _val;
    }

    void setDrawFilteredNormaslStatus(bool _val) {
        draw_filtered_normals_ = _val;
    }

    void setNormals(std::vector<TriMesh::Normal> &_normals_show) {
        normals_show_ = _normals_show;
    }

	void setValues(std::vector<double> &values) {
		values_ = values;
	}

    void RenderBone(float x0, float y0, float z0, float x1, float y1, float z1, double width = 20, bool isCone = false)
    {
        GLdouble  dir_x = x1 - x0;
        GLdouble  dir_y = y1 - y0;
        GLdouble  dir_z = z1 - z0;
        GLdouble  bone_length = sqrt( dir_x*dir_x + dir_y*dir_y + dir_z*dir_z );
        static GLUquadricObj *  quad_obj = NULL;
        if ( quad_obj == NULL )
            quad_obj = gluNewQuadric();
        gluQuadricDrawStyle( quad_obj, GLU_FILL );
        gluQuadricNormals( quad_obj, GLU_SMOOTH );
        glPushMatrix();
        // 平移到起始点
        glTranslated( x0, y0, z0 );
        // 计算长度
        double  length;
        length = sqrt( dir_x*dir_x + dir_y*dir_y + dir_z*dir_z );
        if ( length < 0.0001 ) {
            dir_x = 0.0; dir_y = 0.0; dir_z = 1.0;  length = 1.0;
        }
        dir_x /= length;  dir_y /= length;  dir_z /= length;
        GLdouble  up_x, up_y, up_z;
        up_x = 0.0;
        up_y = 1.0;
        up_z = 0.0;
        double  side_x, side_y, side_z;
        side_x = up_y * dir_z - up_z * dir_y;
        side_y = up_z * dir_x - up_x * dir_z;
        side_z = up_x * dir_y - up_y * dir_x;
        length = sqrt( side_x*side_x + side_y*side_y + side_z*side_z );
        if ( length < 0.0001 ) {
            side_x = 1.0; side_y = 0.0; side_z = 0.0;  length = 1.0;
        }
        side_x /= length;  side_y /= length;  side_z /= length;
        up_x = dir_y * side_z - dir_z * side_y;
        up_y = dir_z * side_x - dir_x * side_z;
        up_z = dir_x * side_y - dir_y * side_x;
        // 计算变换矩阵
        GLdouble  m[16] = { side_x, side_y, side_z, 0.0,
            up_x,   up_y,   up_z,   0.0,
            dir_x,  dir_y,  dir_z,  0.0,
            0.0,    0.0,    0.0,    1.0 };
        glMultMatrixd( m );
        // 圆柱体参数
        GLdouble radius = width;    // 半径
        //GLdouble radius = width;
        GLdouble slices = 40.0;      //  段数
        GLdouble stack = 3.0;       // 递归次数
        GLfloat topRadius = isCone ? 0.f : radius;
        gluCylinder( quad_obj, radius, topRadius, bone_length, slices, stack );
        glPopMatrix();
    }


    void glDrawCylinder(TriMesh::Point& p0, TriMesh::Point& p1, GLColor color, double width, bool isCone)
    {
        glColor3f(color.r, color.g, color.b);
        RenderBone(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], width, isCone);
    }

	
	void glHSV2RGB(double H, double S, double V, double output[3]) {
		H *= 360;
		double C = S * V;
		double X = C * (1 - abs(fmod(H / 60.0, 2) - 1));
		double m = V - C;
		double Rs, Gs, Bs;
		if (H >= 0 && H < 60) {
			Rs = C;
			Gs = X;
			Bs = 0;
		}
		else if (H >= 60 && H < 120) {
			Rs = X;
			Gs = C;
			Bs = 0;
		}
		else if (H >= 120 && H < 180) {
			Rs = 0;
			Gs = C;
			Bs = X;
		}
		else if (H >= 180 && H < 240) {
			Rs = 0;
			Gs = X;
			Bs = C;
		}
		else if (H >= 240 && H < 300) {
			Rs = X;
			Gs = 0;
			Bs = C;
		}
		else {
			Rs = C;
			Gs = 0;
			Bs = X;
		}

		output[0] = (Rs + m);
		output[1] = (Gs + m);
		output[2] = (Bs + m);
	}

protected:
    TriMesh mesh_show_;

    //add
    std::vector<TriMesh::Normal> normals_show_;
	std::vector<double> values_;


    bool draw_points_status_;
    bool draw_edges_status_;
    bool draw_faces_status_;
    //
    bool draw_normals_status_;
    bool draw_filtered_normals_;

    // compute the bounding box of a mesh
    bool meshBoundingBox(TriMesh::Point &min_coord, TriMesh::Point &max_coord);
};

#endif // MESHEXAMINER_H
