#include <iostream>
#include <fstream>
#include <string>
using namespace std;

enum states{
    watch,
    in_string,
    prepare_in,
    in_comment,
    prepare_out,
    comment_end,
    in_cppcomment
};


int main() {

    //read in the file
    ifstream file;
    file.open("test.cpp",ifstream::binary);
    char* origfile;
    int length;
    if(file.fail()){
        cout<<"reading fail"<<endl;
    }
    else{
        file.seekg (0, file.end);
        length = file.tellg();
        file.seekg (0, file.beg);
        origfile = new char [length];
        file.read (origfile,length);

        file.close();
    }
    //the whold file is stored in origfile as a array of char


    // build a new file
    fstream wfile;
    wfile.open("newfile.cpp",ios::out);
    states current_state = watch;
    string newcontent = "";
    string origcontent = "";


    //iterate every char in orgfile, and use fsm to decide the output
    for(int i = 0; i< length; i++){

        switch (current_state){
        case watch:
            if(origfile[i]=='"'){
                current_state = in_string;
                wfile<<origfile[i];
            }
            else if(origfile[i]=='/'){
                current_state = prepare_in;
                wfile<<origfile[i];
            }
            else{
                wfile<<origfile[i];
            }
            break;
        case prepare_in:
            if(origfile[i]=='*'){
                current_state = in_comment;
                newcontent.push_back('/');
                origcontent.push_back('*');
            }
            else if(origfile[i]=='/'){
                wfile<<origfile[i];
                current_state = in_cppcomment;
            }
            else{
                wfile<<origfile[i];
            }
            break;
        case in_comment:
            if(origfile[i]=='\n'){
                newcontent.push_back('\n');
                newcontent.push_back('/');
                newcontent.push_back('/');
                origcontent.push_back('\n');
            }
            else if(origfile[i]=='*'){
                current_state = prepare_out;
            }
            else{
                newcontent.push_back(origfile[i]);
                origcontent.push_back(origfile[i]);
            }
            break;
        case prepare_out:
            if(origfile[i]=='/'){
                current_state = comment_end;
            }
            else{
                current_state = in_comment;
                newcontent.push_back('*');
                newcontent.push_back(origfile[i]);
                origcontent.push_back('*');
                origcontent.push_back(origfile[i]);
            }
            break;
        case comment_end:
            if(origfile[i]==' '){
                break;
            }
            else if(origfile[i]=='\n'){
                newcontent.push_back('\n');
                wfile<<newcontent;
                current_state = watch;
                newcontent = "";
                origcontent = "";
            }
            else{
                origcontent.push_back('*');
                origcontent.push_back('/');
                wfile<<origcontent;
                wfile<<origfile[i];
                newcontent = "";
                origcontent = "";
                current_state = watch;
            }
            break;
        case in_string:
            if(origfile[i]=='"'){
                current_state = watch;
                wfile<<origfile[i];
            }
            else{
                wfile<<origfile[i];
            }
            break;
        case in_cppcomment:
            if(origfile[i]=='\n'){
                wfile<<'\n';
                current_state = watch;
            }
            else{
                wfile<<origfile[i];
            }
            break;
        }        
    }

    wfile.close();

    return 0;
}