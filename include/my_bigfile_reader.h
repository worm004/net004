#include"stdio.h"
#include"stdlib.h"
#include <inttypes.h>
class MY_BIGFILE_READER{
	public:
	char *buf,*pbuf;
	char nums[30];
	MY_BIGFILE_READER()
	{
		buf = pbuf =0;
	}
	~MY_BIGFILE_READER()
	{
		clear();
	}
	void clear(){
		if (buf)
			delete[]buf;
		pbuf = buf = 0;
	}
	int read(const char *path)
	{
		FILE* file = fopen(path, "r");
		//file->end of the file
		fseek (file,0,SEEK_END);
		//get file length
		long size = ftell (file);
		//file->begin of the file
		rewind (file);
		//new buf
		buf = new char [size+1];
		pbuf= buf;
		//read file to buf
		long n = fread(buf, 1, size, file);
//		if(size != a)
//			size=-1;
		buf[n]=0;
		fclose(file);
		return n;
	}
	void to_str(char *text,char c)
	{
		//copy data from buf to text til c.
		while(*pbuf && (*pbuf!=c) && (*pbuf!='\n'))
			*text++ = *pbuf++;
		*text = 0;
		//remove more c.
		while(*pbuf && (*++pbuf==c))
		if (c == '\n')break;//update 20150414
	}
	float read_float(char c)
	{
		to_str(nums,c);
		return (float)atof(nums);
	}
	int read_int(char c)
	{
		to_str(nums,c);
		return atoi(nums);
	}
	uint64_t read_uint64(char c){//update 20140415
		to_str(nums, c);
		return strtoull(nums, NULL, 10);
	}
};
