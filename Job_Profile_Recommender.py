# importing all the modules from treamlit_model.py file
from streamlit_model import *

def ui_of_job_profile_recommender(JD_Dataframe_copy,JD_Dataframe,cv,vector):
    """this function provides the ui for the project and outputs candidate experience and a dataframe of 
    recommended job profile"""
    
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


