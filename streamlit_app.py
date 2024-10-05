import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import pyarrow.csv as pv
import pyarrow.parquet as pq
import os
import numpy as np

# Set page config
st.set_page_config(page_title="GitHub Insights Dashboard", layout="wide")

# Efficient data loading function
@st.cache_data
def load_data(file_name):
    if file_name.endswith('.csv'):
        return pv.read_csv(file_name).to_pandas()
    elif file_name.endswith('.parquet'):
        return pq.read_table(file_name).to_pandas()

# Load github dataset (smaller file)
github_df = load_data('github_dataset.csv')

# Lazy loading function for repository data
@st.cache_data
def load_repo_data():
    if os.path.exists('repository_data.parquet'):
        return load_data('repository_data.parquet')
    else:
        df = load_data('repository_data.csv')
        df.to_parquet('repository_data.parquet')
        return df

# Preprocess function for language analysis
@st.cache_data
def preprocess_language_data(github_df, repo_df):
    primary_langs = github_df['language'].value_counts().head(10)
    
    langs_used = Counter()
    for langs in repo_df['languages_used'].dropna():
        try:
            lang_list = eval(langs) if isinstance(langs, str) else []
            langs_used.update(lang_list)
        except (SyntaxError, NameError):
            continue
    
    top_langs_used = dict(langs_used.most_common(10))
    
    lang_metrics = github_df.groupby('language').agg({
        'stars_count': 'mean',
        'forks_count': 'mean',
        'repositories': 'count'
    }).sort_values('repositories', ascending=False).head(20)
    
    return primary_langs, top_langs_used, lang_metrics

# Main title
st.title("GitHub Projects Analysis Dashboard")

# Horizontal navigation
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
page = st.radio("", ["Overview", "Language Analysis", "Project Metrics", "Time Analysis"])

if page == "Overview":
    st.header("GitHub Projects Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Projects", f"{len(github_df):,}")
    with col2:
        st.metric("Total Stars", f"{github_df['stars_count'].sum():,}")
    with col3:
        st.metric("Total Forks", f"{github_df['forks_count'].sum():,}")
    with col4:
        st.metric("Total Pull Requests", f"{github_df['pull_requests'].sum():,}")
    
    # Top 10 languages
    lang_dist = github_df['language'].value_counts().head(10)
    fig = px.pie(values=lang_dist.values, names=lang_dist.index, title="Top 10 Primary Languages")
    st.plotly_chart(fig, use_container_width=True)
    
    # Projects with most stars
    top_projects = github_df.nlargest(10, 'stars_count')
    fig = px.bar(top_projects, x='repositories', y='stars_count', title="Top 10 Projects by Stars")
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)

elif page == "Language Analysis":
    st.header("Programming Language Analysis")
    
    with st.spinner('Loading data for Language Analysis...'):
        repo_df = load_repo_data()
        primary_langs, top_langs_used, lang_metrics = preprocess_language_data(github_df, repo_df)
        
        fig = go.Figure(data=[
            go.Bar(name='Primary Language', x=primary_langs.index, y=primary_langs.values),
            go.Bar(name='Languages Used', x=list(top_langs_used.keys()), y=list(top_langs_used.values()))
        ])
        fig.update_layout(title="Primary Languages vs Languages Used", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Bubble chart for language metrics
        fig = px.scatter(lang_metrics, x='repositories', y='stars_count', 
                         size='forks_count', color='forks_count',
                         hover_name=lang_metrics.index,
                         title="Language Popularity and Success Metrics",
                         labels={'repositories': 'Number of Projects', 
                                 'stars_count': 'Average Stars',
                                 'forks_count': 'Average Forks'})
        fig.update_layout(xaxis_type="log", yaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Project Metrics":
    st.header("Project Success Metrics")
    
    # Correlation heatmap
    corr_cols = ['stars_count', 'forks_count', 'issues_count', 'pull_requests', 'contributors']
    corr = github_df[corr_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap of Project Metrics")
    st.plotly_chart(fig, use_container_width=True)
    
    
    # CDF plot for distribution of contributors
    contributors_sorted = np.sort(github_df['contributors'])
    y = np.arange(1, len(contributors_sorted) + 1) / len(contributors_sorted)

    fig = go.Figure(data=go.Scatter(x=contributors_sorted, y=y, mode='lines'))
    fig.update_layout(
        title="Cumulative Distribution of Contributors per Project",
        xaxis_title="Number of Contributors",
        yaxis_title="Cumulative Proportion of Projects",
        xaxis_type="log"  # Use log scale for x-axis to better show the distribution
    )
    fig.update_xaxes(range=[0, np.log10(contributors_sorted.max())])
    st.plotly_chart(fig, use_container_width=True)

elif page == "Time Analysis":
    st.header("Project Creation and Activity Over Time")
    
    with st.spinner('Loading data for Time Analysis...'):
        repo_df = load_repo_data()
        
        # Convert 'created_at' to datetime and ensure it's timezone naive
        repo_df['created_at'] = pd.to_datetime(repo_df['created_at']).dt.tz_localize(None)
        
        # Projects created over time (bar chart)
        projects_over_time = repo_df.resample('Y', on='created_at').size().reset_index()
        projects_over_time.columns = ['Year', 'Projects']
        fig = px.bar(projects_over_time, x='Year', y='Projects',
                     title="Projects Created per Year")
        st.plotly_chart(fig, use_container_width=True)
        
        # Commit activity vs project age (grouped bar chart)
        repo_df['age'] = (pd.Timestamp.now() - repo_df['created_at']).dt.days / 365.25  # age in years
        repo_df['age_group'] = pd.cut(repo_df['age'], bins=[0, 1, 2, 3, 5, 10, 20, 100], 
                                      labels=['<1', '1-2', '2-3', '3-5', '5-10', '10-20', '>20'])
        age_commit_summary = repo_df.groupby('age_group')['commit_count'].agg(['mean', 'median']).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=age_commit_summary['age_group'], y=age_commit_summary['mean'], name='Mean Commits'))
        fig.add_trace(go.Bar(x=age_commit_summary['age_group'], y=age_commit_summary['median'], name='Median Commits'))
        fig.update_layout(title="Project Activity vs Age", 
                          xaxis_title="Project Age (years)", 
                          yaxis_title="Number of Commits",
                          barmode='group')
        st.plotly_chart(fig, use_container_width=True)

# Add a footer with data source information
st.markdown("---")
st.markdown("Data source: GitHub Dataset and Repository Dataset from Kaggle")