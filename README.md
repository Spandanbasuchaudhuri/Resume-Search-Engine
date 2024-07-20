# Resume Skill Extractor

This project preprocesses resumes, extracts skills, and allows for dynamic searching of resumes based on specified skills using the Whoosh library. The script uses multiprocessing for efficient processing and natural language processing (NLP) techniques for text preprocessing.

## Features

- **Resume Preprocessing**: 
  - Cleans and tokenizes the text in resumes.
  - Removes unnecessary characters and spaces.
  - Converts text to lowercase.
  - Uses lemmatization to reduce words to their base form.
  
- **Skill Extraction**: 
  - Extracts predefined skills from the resumes.
  - Dynamically identifies skills present in the resumes.
  - Uses regular expressions to match skills.
  
- **Interactive Search**: 
  - Allows users to search for resumes based on specific skills dynamically.
  - Supports comma-separated input for multiple skills.
  - Ensures intersection search to find resumes containing all specified skills.
  
- **Efficient Indexing**: 
  - Uses the Whoosh library to index and search resumes.
  - Supports updating the index with new skills dynamically.
  - Provides efficient search capabilities using an inverted index.

## How It Works

The process begins with the preprocessing of resumes. The `preprocess_text` function is employed to clean and tokenize the text within the resumes. This function removes unnecessary characters, converts the text to lowercase, and applies lemmatization to reduce words to their base form. This preprocessing step is crucial for ensuring that the text is in a consistent format, making subsequent analysis more accurate.

Once the resumes are preprocessed, the next step involves skill extraction. The `extract_skills` function uses regular expressions to identify predefined skills in the resumes. To enhance efficiency, the `extract_skills_parallel` function applies skill extraction in parallel, leveraging multiple processing cores to handle large datasets more effectively. This parallel processing capability significantly reduces the time required for skill extraction.

Following skill extraction, the resumes and their corresponding skills are indexed using the Whoosh library. The indexing process involves creating an inverted index that allows for efficient searching of resumes based on the extracted skills. The `batch_write` function is used to write the preprocessed resumes and extracted skills to the index in batches, ensuring that the indexing process is both efficient and scalable.

The final phase of the process is the interactive search functionality. The script prompts the user to enter skills for searching. These skills are used to query the index, and the `search_resumes` function searches the index for resumes that contain all the specified skills. The results are then returned as matching resume IDs, enabling users to quickly identify resumes that meet their specific criteria.

This comprehensive approach ensures that resumes are accurately processed and that skills are efficiently extracted and indexed, providing users with a powerful tool for searching resumes based on specific skills.
