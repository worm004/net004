#ifndef BLOB_H
#define BLOB_H
struct Blob{
	public:
	~Blob();
	void set_shape(const Blob& b);
	void set_shape(int n, int c, int h, int w);
	void set_data(float * data);
	void alloc();
	void clear();
	void show() const;
	int total();

	public:
	float *data = 0;
	int n = 0, c = 0, w = 0, h = 0;
	
	private:
	bool owner = true;
};
#endif
