#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <GLUT/glut.h>

// for all
int windowWidth		= 800;
int windowHeight	= 600;

// for OpenCV
const char* CV_WINDOW_NAME			= "Vision";
const char* CV_CASCADE_NAME			= "haarcascade_frontalface_alt_tree.xml";
CvCapture * camera					= NULL;
CvHaarClassifierCascade * cascade	= NULL;
CvMemStorage * storage				= NULL;
float faceX							= 0;
float faceY							= 0;

// for OpenGL
const char* GL_WINDOW_NAME	= "Graphics";
GLfloat lightAmbient[]		= {1.0, 1.0, 1.0, 1.0};		// 环境光
GLfloat lightDiffuse[]		= {1.0, 1.0, 1.0, 1.0};		// 散射光
GLfloat lightPosition[]		= {0.0, 0.0, 0.0, 1.0};		// 光源坐标
int glWindow = 0;


void init(int argc, const char * argv[]);
void detectPosition();
void drawObject(GLfloat x, GLfloat y, GLfloat z, GLdouble size);
void display(void);
void reshape(int w, int h);
void timer(int value);
void done();
void keyboard(unsigned char key, int x, int y);

int main(int argc, const char * argv[]) {
	init(argc, argv);
	glutMainLoop();
	done();
	
    return 0;
}

void init(int argc, const char * argv[]) {
	// init for CV
	cvNamedWindow(CV_WINDOW_NAME, CV_WINDOW_AUTOSIZE);
	camera	= cvCreateCameraCapture(CV_CAP_ANY);
	cascade	= (CvHaarClassifierCascade *) cvLoad(CV_CASCADE_NAME, 0, 0, 0);
	storage	= cvCreateMemStorage(0);
	
	// init for GL
	glutInit(& argc, (char **) argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(windowWidth, windowHeight); 
	glutInitWindowPosition(0, 0);
	glWindow = glutCreateWindow(GL_WINDOW_NAME);
	
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glutDisplayFunc(display); 
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutTimerFunc(1, timer, 0);
	
	// 光照设置
	glEnable(GL_LIGHTING);								// 打开光照
	glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);		// 设置光照0的环境光
	glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);		// 设置光照0的散射光
	glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);	// 放置光照0
	glEnable(GL_LIGHT0);								// 打开光照0
	
	glEnable(GL_CULL_FACE);								// 打开背面裁剪
	glEnable(GL_DEPTH_TEST);							// 打开深度检测
}

void drawObject(GLfloat x, GLfloat y, GLfloat z, GLdouble size) {
	glPushMatrix();
	
	x = x - faceX * windowWidth * 0.5;
	y = y - faceY * windowHeight * 0.5;
	GLfloat rx = cvFastArctan(y, - z);
	GLfloat ry = - cvFastArctan(x, - z);
	GLfloat rz = 0;
	
	glRotatef(rx, 1, 0, 0);
	glRotatef(ry, 0, 1, 0);
	glRotatef(rz, 0, 0, 1);
	glTranslatef(x, y, z);
	
	glutWireSphere(size, 16, 16);
	//glutSolidCube(size);
	//glutWireTeapot(size);
	
	glPopMatrix();
}

void display(void) {	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	
	
	detectPosition();
	
	drawObject(0, 0, -1000, 100);
	
	drawObject(-87, 50, -400, 50);
	drawObject(-87, 50, -800, 50);
	drawObject(-87, 50, -1200, 50);
	drawObject(-87, 50, -1600, 50);
	drawObject(-87, 50, -2000, 50);
	
	drawObject(87, 50, -400, 50);
	drawObject(87, 50, -800, 50);
	drawObject(87, 50, -1200, 50);
	drawObject(87, 50, -1600, 50);
	drawObject(87, 50, -2000, 50);
	
	drawObject(0, -100, -400, 50);
	drawObject(0, -100, -800, 50);
	drawObject(0, -100, -1200, 50);
	drawObject(0, -100, -1600, 50);
	drawObject(0, -100, -2000, 50);
	
	glutSwapBuffers();									// 交换双缓冲区
}

// detect face position
void detectPosition() {	
	IplImage * frame			= cvQueryFrame(camera);
	cvFlip(frame, NULL, 1);
	IplImage * grayImage		= cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	IplImage * forShowImage		= cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 3);
	
	cvCvtColor(frame, grayImage, CV_BGR2GRAY);
	cvCopy(frame, forShowImage, NULL);

	CvSeq * faces = cvHaarDetectObjects(grayImage, cascade, storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(120, 120), cvSize(240, 240));
	for (int i = 0; i < faces->total; i++) {
		CvRect * rect = (CvRect *) cvGetSeqElem(faces, i);
		cvRectangle(forShowImage, 
					cvPoint(rect->x, rect->y), 
					cvPoint(rect->x + rect->width, rect->y + rect->height), 
					CV_RGB(0, 255, 0), 
					3, 8, 0);
		faceX = (float) (rect->x + rect->width / 2) / (float) frame->width - 0.5;
		faceY = 0.5 - (float) (rect->y + rect->height / 2) / (float) frame->height;
	}
	cvShowImage(CV_WINDOW_NAME, forShowImage);
	
	cvClearMemStorage(storage);
	cvReleaseImage(& forShowImage);
	cvReleaseImage(& grayImage);
}

void reshape(int width, int height) {
	glViewport(0, 0, (GLsizei) width, (GLsizei) height);				// 视口大小
	glMatrixMode(GL_PROJECTION);										// 透视投影
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat) width / (GLfloat) height, 1, 10240);	// 投影参数
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void timer(int value) {
	display();
	glutTimerFunc(20, timer, 0);
}

void done() {
	cvReleaseCapture(& camera);
	cvReleaseHaarClassifierCascade(& cascade);
	cvReleaseMemStorage(& storage);
	cvDestroyWindow(CV_WINDOW_NAME);
	glutDestroyWindow(glWindow);
}

void keyboard(unsigned char key, int x, int y) {
	switch (key) {
		case 'q':
			exit(0);
			break;
		default:
			break;
	}
}
