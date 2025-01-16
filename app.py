import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from imdb import IMDb

# Initialize IMDb
ia = IMDb()

# Streamlit UI
st.title("🎥 Similar Movie Recommendation Assistant")
st.subheader("Provide Movie Details")

# Ask user for their Groq API key
groq_api_key = st.text_input("Enter your Groq API Key", type="password")

# Proceed only if the API key is provided
if groq_api_key:
    # Initialize Groq API with the provided key
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

    # Initialize the prompt template for the synopsis-based recommendation
    synopsis_prompt_template = ChatPromptTemplate.from_template("""
        You are a knowledgeable movie recommendation assistant specializing in global cinema.
        Based on the user's provided movie's synopsis, find 3 similar movies from the entire IMDb database, 
        including their titles, years, countries of origin, storylines, IMDb ratings, and explain why each of the movies is similar to the provided movie.
        
        Movie Synopsis:
        {storyline}
        
        Movie IMDb Rating:
        {rating}
        
        Your Response:
        Provide a list of 3 movies with their titles, years, countries, IMDb ratings, storylines, and explain the similarity of each movie with the provided movie's storyline.
    
        For each movie, explain why it was recommended and how its storyline or themes are similar to the given movie.
        After providing the explanation, leave a line break for clarity. Make sure the storyline and explanation are visually separated.
    """)

    # Input for movie name and year
    movie_name = st.text_input("Enter Movie Name")
    movie_year = st.number_input("Enter Movie Year", min_value=1900, max_value=2024, step=1)

    if st.button("Get Similar Movie Recommendations"):
        if not movie_name or not movie_year:
            st.error("Please provide both movie name and year.")
        else:
            # Search IMDb for the movie by name and year
            search_results = ia.search_movie(movie_name)
            matching_movie = None

            # Find the movie with the exact match for year
            for movie in search_results:
                if movie.get('year') == movie_year:
                    matching_movie = ia.get_movie(movie.movieID)
                    break

            if not matching_movie:
                st.error("No matching movie found for the given name and year.")
            else:
                # Fetch the movie's synopsis or plot and IMDb rating
                movie_synopsis = matching_movie.get('synopsis')
                movie_plot = matching_movie.get('plot')
                movie_rating = matching_movie.get('rating')

                # If no synopsis, use plot as fallback for movie storyline
                if movie_synopsis:
                    movie_storyline = movie_synopsis
                elif movie_plot:
                    movie_storyline = movie_plot[0]  # Use the first plot if multiple available
                else:
                    movie_storyline = "No storyline available."

                if not movie_rating:
                    movie_rating = "Not available"  # Default text if no rating is found

                # Step 1: If synopsis is available, recommend movies based on synopsis
                if movie_synopsis:
                    user_input_synopsis = {
                        "storyline": movie_synopsis,
                        "rating": movie_rating,
                    }
                    prompt_synopsis = synopsis_prompt_template.format(**user_input_synopsis)

                    try:
                        # Call Groq for similar movie recommendations based on synopsis
                        response_synopsis = llm.invoke(prompt_synopsis)

                        if hasattr(response_synopsis, "content"):
                            response_content_synopsis = response_synopsis.content.strip()
                            if response_content_synopsis:
                                st.subheader("Similar Movie Recommendations (Based on Synopsis):")
                                st.write(response_content_synopsis)
                            else:
                                st.write("No similar movies found based on the provided synopsis.")
                        else:
                            st.error("The response does not contain text content from synopsis-based recommendation.")
                    except Exception as e:
                        st.error(f"An error occurred while fetching similar movies based on synopsis: {str(e)}")
                
                # Step 2: Always suggest based on plot (if available)
                if movie_plot and not movie_synopsis:
                    user_input_plot = {
                        "storyline": movie_plot[0],  # Use the first plot if multiple available
                        "rating": movie_rating,
                    }
                    prompt_plot = synopsis_prompt_template.format(**user_input_plot)

                    try:
                        # Call Groq for similar movie recommendations based on plot
                        response_plot = llm.invoke(prompt_plot)

                        if hasattr(response_plot, "content"):
                            response_content_plot = response_plot.content.strip()
                            if response_content_plot:
                                st.subheader("Similar Movie Recommendations (Based on Plot):")
                                st.write(response_content_plot)
                            else:
                                st.write("No similar movies found based on the provided plot.")
                        else:
                            st.error("The response does not contain text content from plot-based recommendation.")
                    except Exception as e:
                        st.error(f"An error occurred while fetching similar movies based on plot: {str(e)}")

else:
    st.info("Please enter your Groq API key to continue.")
