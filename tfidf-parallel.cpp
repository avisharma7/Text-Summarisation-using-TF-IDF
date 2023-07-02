#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

// Regular expression for tokenization
const regex re(R"([\s|,]+)");
const regex sre(R"([\\.][\s]+)");

vector<string> tokenize(const string str, const regex re)
{
    sregex_token_iterator it{ str.begin(), str.end(), re, -1 };
    vector<std::string> tokenized{ it, {} };
 
    // Additional check to remove empty strings
    tokenized.erase(remove_if(tokenized.begin(),
                            tokenized.end(),
                            [](string const& s) {
                            return s.size() == 0;
                            }), 
                    tokenized.end()
    );
 
    return tokenized;
}
 //sanitisation
 //removal of other characters except alpha-numeric
 //highercase to lower case

void sanitize(string &token){
    if(!iswalnum(token[token.size() - 1])) {
        token.pop_back();
    }
    transform(token.begin(), token.end(), token.begin(), ::tolower);
}

map<string, int> termFrequency(vector<string> &tokens){
    map<string, int> tf;
    #pragma omp parallel for
    for(int i=0;i<tokens.size();i++){
        sanitize(tokens[i]);
        #pragma omp critical 
        // to avoid race condition 
// critical construct is used to synchronize access to shared resources
 // such as the priority queue that stores the top sentences.
        ++tf[tokens[i]];
    }
    return tf;
}

double calculateScore(string sentence, map<string, int> &tf, map<string, int> &idf){
    vector<string> words = tokenize(sentence, re);
    double score = 0;
    #pragma omp parallel for reduction(+: score)
    for(int i=0;i<words.size();i++)
    {
        if(!idf[words[i]]) continue;
        score += 100.0 * tf[words[i]] / idf[words[i]];
    }
    return score;
}


int main(){
    omp_set_num_threads(12);
    //setting num of threads that used by openmp

    cout << "-----------------------------------" << '\n';
    cout << "-    Parallel Summarization       -" << '\n';
    cout << "-----------------------------------\n" << '\n';

    // Start time
    double t1 = omp_get_wtime();

    string fname="input.txt";

    // Reading the input file
    fstream fin(fname);
    string text="";
    string line;
    while(getline(fin, line)){
        text += line + ' ';
    }
    fin.close();
       
    // Word Tokenization 
    vector<string> tokenized = tokenize(text, re);
    
    // Sentence Tokenization 
    vector<string> sentences = tokenize(text, sre);

    // Finding term frequencies
    map<string, int> tf = termFrequency(tokenized);
    
    // Generating inverse document frequency
    fstream model("model.txt");

    map<string, int> idf = tf;

    string word;
    int frequency;

    while(model >> word >> frequency){
        idf[word] += frequency;
    }
    
    model.close();

    double ratio = 1.0/3;
    int summarySize = ceil(ratio * sentences.size()); 
    
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;

    // cout << sentences.size() << '\n';
    // Ranking sentences
    #pragma omp parallel for
    for(int i=0; i<sentences.size(); ++i)
    {
        double score = calculateScore(sentences[i], tf, idf);
    // critical construct is used to synchronize access to shared resources
    // such as the priority queue that stores the top sentences.
        #pragma omp critical 
        {
            if(pq.size() == summarySize && pq.top().first <score) 
        {
            pq.pop();
            pq.push({score, i});
        }
        else if(pq.size()<summarySize)
        {
            pq.push({score, i});
        }
        }
    }
    vector<pair<double, int>> vec;
    while (!pq.empty()) {
        vec.push_back(pq.top());
        pq.pop();
    }
    sort(vec.begin(), vec.end(), [](const auto& left, const auto& right) {
        return left.second < right.second;
    });


    fstream fout("output.txt");
    for (int i = 0; i < vec.size(); i++) {
    int index = vec[i].second;
    fout << sentences[index] << '\n';
}
    fout.close();

    // Updating the inverse document frequency
    // #pragma omp parallel for 
    fstream updateModel("model.txt");
    for(auto &mod: idf)
    {
        updateModel << mod.first << ' ' << mod.second << '\n';
    }
    updateModel.close();

    double t2 = omp_get_wtime();

    cout << "- Summarization Completed -" << '\n';
    cout << "- Time Taken : " << t2-t1 << " s -\n";

    return 0;
}