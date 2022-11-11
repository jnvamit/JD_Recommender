import pandas as pd
import numpy as np
import re
from functools import reduce
from PyPDF2 import PdfFileReader
import docx2txt
import streamlit as st

def initialising():
    """this function is the starting point of the project, where the dataframes are read,
    and countvectorizer and cosine_similarity are used.

    parameter returned:
        JD_Dataframe_copy:initial dataframe having multiple columns
        JD_Dataframe: processed dataframe having only  two columns
        cv: countvectorizer object
        vector: this object stores an array of vectorized value of 'Job_Description'
    """

    JD_Dataframe_copy = pd.read_csv('JD_Dataframe_final_multicolumn_final')
    JD_Dataframe = pd.read_csv('JD_Dataframe_two_column_final')
    # JD_Dataframe.drop_duplicates(inplace=True,ignore_index=True)
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=5000,stop_words='english')
    # Using countVectorizor for Job_Description Column
    vector = cv.fit_transform(JD_Dataframe['Job_Description']).toarray()
    # Using cosine-similarity
    from sklearn.metrics.pairwise import cosine_similarity
    # Aplying cosine-similarity on resultant vector
    similarity = cosine_similarity(vector)
    return JD_Dataframe_copy,JD_Dataframe,cv,vector


def read_pdf(file):
    """this function is used to read content of pdf files.

    input parameter:
        file: object for pdf file content.

    parameter returned:
        all_pages_text: text format of pdf content.

    """

    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_pages_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_pages_text += page.extractText()
    return all_pages_text  

def text_preprocessing(string):

    """this function is used to format texual content.

    input parameter:
        string: un-formatted texual content.

    parameter returned:
        string: formatted texual content.

    """

    string = string.lower()
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    chars_to_remove = ["|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('\n', ' ')
    string = string.replace(',', ' ')
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    # string = string.replace('-', ' ')
    return string     

# Dataframe creation

def making_dataframe(res,JD_Dataframe):

    """this function is used to make dataframe.

    input parameter:
        res: resume content.
        JD_Dataframe: Dataframe containing whole JD information.

    parameter returned are:
        df: A dataframe in which the resume content has been concatenated.

    """

    dictt = [['Resume_1',res]]
    df = pd.DataFrame(dictt,columns=['Job_Title','Job_Description'])
    JD_Dataframe.loc[len(JD_Dataframe.index)] = ['resume2',res]
    return df

# transforming the resume content 

def transform(data_frame,cv,vector):

    """this function is transform the content of the dataframe using countvectorizer.

    input parameter:
        data_frame: its the dataframe to be vectorized.
        cv: countvectorizer object.
        vector: initial vectorized data.

    parameter returned are:
        temp: array of vectorized data.

    """

    vec = cv.transform(data_frame['Job_Description']).toarray()
    temp = np.append(vector,vec,axis = 0)
    return temp    


def experience_returner(index_list,content):

    """this function returns the associated experience of a resume.

    input parameter:
        index_list: list containing items, having the term 'experience' associated with it.
        content: resume content.

    parameter returned are:
        float value of experience..

    """

    for i in index_list:
        p = content[i-30:i+30]
        q =  re.findall(r"\d*\.\d+|\d+", p)
        q = [float(num) for num in q if float(num) < 30]
        if q:
            if len(q) > 1:
                return float(q[0]+1)
            else:
                return float(q[0])
        else:
            pass    

def experience_extractor(resume_content):

    """this function is extract numbers surronding the term 'experience'.

    input parameter:
        resume_content: texual data of resume.

    parameter returned are:
        index_list: list containing items, having the term 'experience' associated with it.
        resume_content: resume content.

    """

    matcher = re.finditer('experience',resume_content)
    index_list = []
    for i in matcher:
        index_list.append(i.start())
    return experience_returner(index_list,resume_content)  

def proper_ranking_value_dataframe(indexes,Dicts,experience,JD_Dataframe_copy):

    """this function ranks the return recommendation.

    input parameter:
        indexes: recommended indexes.
        experience: extracted experience.
        JD_Dataframe_copy: dataframe having jd information.

    parameter returned are:
        df : dataframe of recommended JD.

    """

    condition_satisfied_arr = []
    condition_not_satisfied_arr = []
    output_indexes_satisfied = []
    output_indexes_not_satisfied = []
    Dict={}
    dictt = {'File_name':[],'Job_Title':[],'Similarity':[],'Experience':[]}
    df = pd.DataFrame(dictt)
    for i in indexes:
        if experience>= JD_Dataframe_copy.Lower_Exp[i] and experience<= JD_Dataframe_copy.Higher_Exp[i] :
            condition_satisfied_arr.append(JD_Dataframe_copy.Job_Title[i]) 
            output_indexes_satisfied.append(i)
        else:
            condition_not_satisfied_arr.append(JD_Dataframe_copy.Job_Title[i])
            output_indexes_not_satisfied.append(i)

    for i in output_indexes_not_satisfied:
        Dict[i] = abs(JD_Dataframe_copy.Lower_Exp[i])+abs(JD_Dataframe_copy.Higher_Exp[i])
        
    p= dict(sorted(Dict.items(), key=lambda item: item[1],reverse=False))  
    output_indexes_not_satisfied = list(p.keys())
    output  =  condition_satisfied_arr+condition_not_satisfied_arr,output_indexes_satisfied+output_indexes_not_satisfied
    for i in range(len(output[0])):
        j_t = output[0][i]
        txt1 = j_t.split()
        x = [i.capitalize() for i in txt1]
        j_t = " ".join(x)
        l_e = JD_Dataframe_copy.Experience[output[1][i]]
        f_n = JD_Dataframe_copy.File_name[output[1][i]]
        sim = round(Dicts[output[1][i]]*100)
        df.loc[len(df.index)] = [f_n,j_t,str(sim)+"%",l_e]
    df.index = np.arange(1, len(df) + 1)    
    return df 
    # return np.concatenate((condition_satisfied_arr, condition_not_satisfied_arr), axis = 0)         

def jd_matcher_dataframe(reading_cv,JD_Dataframe_copy,JD_Dataframe,cv,vector):

    """this function recommends the JD for a given resume.

    input parameter:
        reading_cv: resume content
        JD_Dataframe_copy: multi-column dataframe
        JD_Dataframe: two-column dataframe
        cv: object of countvectorizer
        vector: vectorized formed of JD data

    parameter returned are:
        df : dataframe of recommended JD.

    """

    from sklearn.metrics.pairwise import cosine_similarity
    experience = experience_extractor(reading_cv)
    processed_resume = text_preprocessing(reading_cv)
    data_frame = making_dataframe(processed_resume,JD_Dataframe)
    vec = transform(data_frame,cv,vector)
    similarity = cosine_similarity(vec)
    distances = sorted(list(enumerate(similarity[-1])),reverse=True,key = lambda x: x[1])
    Job_Titles = []
    Job_Indexes = []
    cosine_sim = []
    Dict = {}
    for i in distances[1:6]:
      cosine_sim.append(i) 
      Job_Indexes.append(i[0])
      Job_Titles.append(JD_Dataframe_copy.Job_Title[i[0]])
      # print(JD_Dataframe_copy.Job_Title[i[0]],'\t',JD_Dataframe_copy.Experience[i[0]])
    JD_Dataframe.drop(JD_Dataframe.index[len(JD_Dataframe):], inplace=True) 

    for i in cosine_sim:
      Dict[i[0]]=i[1]

    if experience:
        st.write("Job Profile Recommendation is based on experience and skills")
        return proper_ranking_value_dataframe(Job_Indexes,Dict,experience,JD_Dataframe_copy)
    else:
        st.write("Could not detect the experience. Job Profile Recommendation is based on skills")
        dictt = {'File_name':[],'Job_Title':[],'Similarity':[],'Experience':[]}
        df = pd.DataFrame(dictt)
        for i in range(len(Job_Titles)):
            j_t = Job_Titles[i]
            sim = round(Dict[Job_Indexes[i]]*100)
            l_e = JD_Dataframe_copy.Experience[Job_Indexes[i]]
            f_n = JD_Dataframe_copy.File_name[Job_Indexes[i]]
            txt1 = j_t.split()
            x = [i.capitalize() for i in txt1]
            j_t = " ".join(x)
            df.loc[len(df.index)] = [f_n,j_t,str(sim)+"%",l_e]
        df.index = np.arange(1, len(df) + 1)    
        return df          

def ui_of_jd_recommender(JD_Dataframe_copy,JD_Dataframe,cv,vector):

    """this function creates the UI for the project.

    input parameter:
        JD_Dataframe_copy: multi-column dataframe
        JD_Dataframe: two-column dataframe
        cv: object of countvectorizer
        vector: vectorized formed of JD data

    """

    st.markdown('Job Profile Recommender Project')
    file_input = st.file_uploader("Select the Resume",type=['pdf','docx','doc'])
    if st.button("Recommend"):
        if file_input is not None:
            if file_input.type == "application/pdf":
                raw_text = read_pdf(file_input)
            else:
                raw_text = docx2txt.process(file_input)
            
            file_details = {
                                "Candidate_Experience":experience_extractor(raw_text)
                            }
            st.write("Candidate Experience: ",file_details["Candidate_Experience"]) 
            st.write(jd_matcher_dataframe(raw_text,JD_Dataframe_copy,JD_Dataframe,cv,vector))
            
if __name__ == "__main__":
    JD_Dataframe_copy,JD_Dataframe,cv,vector = initialising()
    ui_of_job_profile_recommender(JD_Dataframe_copy,JD_Dataframe,cv,vector)            
            
            
            
            
