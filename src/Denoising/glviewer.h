#ifndef GLVIEWER_H
#define GLVIEWER_H

#include <QtOpenGL>
#include "meshexaminer.h"

#include <QColorDialog>

class GLViewer : public QGLWidget
{
    Q_OBJECT

public:
    GLViewer(QWidget *parent = 0);
    ~GLViewer();

    void updateMesh(const TriMesh &_mesh);
    void resetMesh(const TriMesh &_mesh, bool _needNormalize = false);

    MeshExaminer* getExaminer(){
        return examiner_;
    }

public slots:
    void setDrawPointsStatus(bool _val){
        examiner_->setDrawPointsStatus(_val);
        this->updateGL();
    }

    void setDrawFacesStatus(bool _val){
        examiner_->setDrawFacesStatus(_val);
        this->updateGL();
    }

    void setDrawEdgesStatus(bool _val){
        examiner_->setDrawEdgesStatus(_val);
        this->updateGL();
    }

    void setDrawNormalStatus(bool _val) {
        examiner_->setDrawNormaslStatus(_val);
        this->updateGL();
    }


    void setDrawFilteredNormalsStatus(bool _val) {
        examiner_->setDrawFilteredNormaslStatus(_val);
        this->updateGL();
    }

    void setFilteredNormals(std::vector<TriMesh::Normal> filtered_normals) {
         examiner_->setNormals(filtered_normals);
    }

	void setValueToShow(std::vector<double> values) {
		examiner_->setValues(values);
	}

    void setBackgroundColor(){
        QColor color = QColorDialog::getColor(Qt::black, this, tr("Set Background Color!"));
        if(!color.isValid()) return;
        this->qglClearColor(color);
        this->updateGL();
    }

protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);

    void initializeGL();
    void resizeGL(int _w, int _h);
    void paintGL();

private:
    MeshExaminer *examiner_;
};

#endif // GLVIEWER_H
