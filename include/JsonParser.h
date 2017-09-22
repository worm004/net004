#ifndef JSONPARSER_H
#define JSONPARSER_H
#include <string>
#include <vector>
#include <queue>
#include <map>
/* reference: http://www.json.org */
struct JsonPrimitiveValue{
	//string/num
	std::string type;
	std::string s;
	double d;
	std::string to_str();
};
class JsonValue{
	public:
	JsonValue();
	JsonValue(const std::string& type);
	JsonValue(const std::string& type, double d);
	JsonValue(const std::string& type, const std::string& s);
	void set_array(const char* b, int n,std::queue<int>& helper);
	void set_obj(const char*b, int n,std::queue<int>& helper);
	std::string to_str(int level);

	public:
	std::map<std::string, JsonValue> jobj;
	std::vector<JsonValue> jarray;
	JsonPrimitiveValue jv;
	//obj/array/v/null
	std::string type;

	private:
	void set_val(const std::string& v);
	void set_null();
	std::string check_type(const char*b, const char*e,int& ib, int &ie,bool has_colon);
	void set_by_type(const std::string& type, JsonValue& obj, const char*& c, const char*e, int val_b, int val_e, std::queue<int>& helper);
};
class JsonParser{
	public:
	void read(const std::string& path);
	void write(const std::string& path);
	void show();
	private:
	void gen_helper(const char*b, const char*e);
	
	public:
	JsonValue j;
	private:
	std::queue<int> helper;
};
#endif
