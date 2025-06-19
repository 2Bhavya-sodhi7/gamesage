import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import numpy as np
from collections import Counter
from PIL import Image
from wordcloud import WordCloud
from textblob import TextBlob
from matplotlib.patches import Wedge
import matplotlib.patches as patches


# Automatically load datasets


@st.cache_data
def load_data():
    sponsor_df = pd.read_csv("final_sponsor_detections(3).csv")
    audio_df = pd.read_csv("final_match_sponsor_data_colab (1).csv")
    df1 = pd.read_csv("cricket_shots.csv")
    df2 = pd.read_csv("IPL2k24_tweets_data.csv")  
    df3 = pd.read_csv("stadium_boundaries.csv")  # Assuming this is the physical benchmarks data
    return sponsor_df, audio_df, df1, df2, df3

# Load data
try:
    sponsor_df, audio_df, df1, df2, df3 = load_data()

    # Title and header
    st.title('GameSage: play the game beyond the game')
    st.header('(Missed Branding opportunities):')


       





        # Section 4: Match Moments Sponsor Count
    st.markdown("## 1. Match Moments Sponsor Count")
    st.write("The graph shows that during important parts of a match, not all sponsors were visible, meaning they lost out on chances to be seen by the audience.(based on audio peaks).")

    # Prepare data: split sponsors and count per timestamp
    sponsor_count_df = audio_df[["TimestampFrameNumber", "VisibleSponsorsDuringPeak"]].copy()
    sponsor_count_df["SponsorList"] = sponsor_count_df["VisibleSponsorsDuringPeak"].str.split(", ")
    sponsor_count_df["SponsorCount"] = sponsor_count_df["SponsorList"].apply(
        lambda x: 0 if x == ["NoSponsorDetected"] else len(x)
    )

    # Filter for moments where at least 1 sponsor was visible
    filtered_df = sponsor_count_df[sponsor_count_df["SponsorCount"] > 0]

    # Bubble chart: Timestamp vs. SponsorCount (size = count)
    st.subheader("Sponsor Count at Key Match Moments")
    fig_bubble, ax_bubble = plt.subplots(figsize=(10, 4))
    ax_bubble.scatter(
        filtered_df["TimestampFrameNumber"],
        filtered_df["SponsorCount"],
        s=filtered_df["SponsorCount"] * 50,  # scale bubble size
        alpha=0.6,
        color="mediumvioletred"
    )
    ax_bubble.set_xlabel("Timestamp (Frame Number)")
    ax_bubble.set_ylabel("Sponsor Count")
    ax_bubble.set_title("Number of Sponsors Visible at Key Match Moments")
    st.pyplot(fig_bubble)






    # Section 2: Visible Sponsors & Asset Types
    st.markdown("## 2. Visible Sponsors & Asset Types")
    st.write("The chart shows that a significant portion of sponsor placements were either blurry (15.5) or obstructed(8.9%).)")
    st.write("This means even when a sponsor's logo was present, it wasn't clearly visible to the audience, resulting in missed opportunities for brand recognition and exposure.")


    
    st.dataframe(sponsor_df)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sponsor-wise Asset Count")
        sponsor_counts = sponsor_df["sponsor_name"].value_counts()
        st.bar_chart(sponsor_counts)

    with col2:
        st.subheader("Asset Type Distribution")
        asset_counts = sponsor_df["sponsor_asset_type"].value_counts()
        st.bar_chart(asset_counts)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Visibility Breakdown")
        fig_vis, ax_vis = plt.subplots()
        vis_counts = sponsor_df["sponsor_asset_visibility"].value_counts()
        ax_vis.pie(vis_counts, labels=vis_counts.index, autopct='%1.1f%%', startangle=90)
        ax_vis.axis("equal")
        st.pyplot(fig_vis)

    with col4:
        st.subheader("Confidence Score Distribution")
        fig_conf, ax_conf = plt.subplots()
        sns.histplot(sponsor_df["confidence"], bins=20, kde=True, color="steelblue", ax=ax_conf)
        ax_conf.set_xlabel("Confidence Score")
        st.pyplot(fig_conf)

    st.subheader("Top 10 Sponsors by Asset Count")
    top10_df = sponsor_counts.head(10).reset_index()
    top10_df.columns = ["Sponsor", "Count"]
    st.table(top10_df)

    # Section 3: Peak Audio Score Analysis
    st.markdown("## 3. Peak Audio Score Analysis")
    st.write("During audio peak moments, which indicate high-attention instances, several sponsors are detected very few times or not at all ")
    st.write("This signifies a missed branding opportunity as these sponsors are not gaining visibility during the most impactful and engaging parts of the match broadcast  ")
    st.subheader("Raw Audio Peak Data Table")
    st.dataframe(audio_df)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Audio Peak Score Distribution")
        fig_peak, ax_peak = plt.subplots()
        sns.histplot(audio_df["AudioPeakScore"], bins=20, kde=True, color="teal", ax=ax_peak)
        ax_peak.set_xlabel("AudioPeakScore")
        st.pyplot(fig_peak)

    with col6:
        st.subheader("Sponsors Detected at Audio Peaks")
        exploded = audio_df["VisibleSponsorsDuringPeak"].str.split(", ").explode()
        peak_counts = exploded.value_counts().drop("NoSponsorDetected", errors="ignore")
        st.bar_chart(peak_counts)

except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}. Make sure your CSV files are in the folder.")


st.header('(Power of Prediction and Analyis):')
st.header("4: Map targeting")
st.write("Analyzing these hotspots lets organizations understand fan behavior, so they can target marketing campaigns and improve fan experiences in those specific areas .")
components.iframe(
    "https://sponsor-map-33vc.vercel.app",
    height=600,
    width=1000,
    scrolling=True
)
st.header("Stadium Analysis")

st.subheader("Raw Ball-by-Ball Data (RCB VS PBKS FINAL 2025)")
st.dataframe(df1, use_container_width=True)

# Assume df1 is already loaded via load_data()
# df1 = pd.read_csv("cricket_shots.csv")  # already executed in your load_data()

st.subheader("Shot-Direction Frequency:")
st.write("This chart uses historical data to analyze and predict where shots are most frequently hit in a stadium, showing patterns that would otherwise be hidden .")

# 1. Count occurrences of each shot direction
shot_counts = df1['shot_direction'].value_counts().sort_index()  # pandas.Series.value_counts()[1]

# 2. Render a simple bar chart in Streamlit
st.bar_chart(shot_counts)  # Streamlit bar_chart API[2]


shot_counts = df1['shot_direction'].value_counts().sort_index()
shot_avg_runs = df1.groupby('shot_direction')['runs'].mean().loc[shot_counts.index]

# Streamlit visualization
st.subheader("Average Runs by Shot Direction")
st.write("By understanding these likely shot directions, brands can strategically place advertisements in high-visibility areas, maximizing their exposure and return on investment .")
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar chart: counts
sns.barplot(x=shot_counts.index, y=shot_counts.values, ax=ax1, palette='Blues')
ax1.set_xlabel("Shot Direction")
ax1.set_ylabel("Delivery Count", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticklabels(shot_counts.index, rotation=45, ha='right')

# Line chart: average runs
ax2 = ax1.twinx()
sns.lineplot(x=shot_avg_runs.index, y=shot_avg_runs.values, marker='o', color='red', ax=ax2)
ax2.set_ylabel("Average Runs", color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax1.set_title("Shot Direction Frequency and Average Runs") 
st.pyplot(fig)

st.subheader("Shots per Over (0–19)")

# Compute shot counts by over in chronological order
shots_per_over = df1['over'].value_counts().sort_index()

# Render interactive line chart in Streamlit
st.line_chart(shots_per_over)

# Optional: Customized Matplotlib line plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(shots_per_over.index, shots_per_over.values, marker='o', color='tab:blue')
ax.set_xlabel("Over Number")
ax.set_ylabel("Number of Shots")
ax.set_title("Total Shots per Over Across Innings")
ax.set_xticks(range(0, 20))
ax.grid(alpha=0.3)
st.pyplot(fig)

coord_map = {
    # Straight shots
    "Straight": (0.5, 0.5),
    "Straight Long On": (0.5, 0.8),
    "Straight Mid Off": (0.5, 0.3),
    
    # On side (leg side) positions
    "Long On": (0.4, 0.9),
    "Mid-Wicket": (0.2, 0.6),
    "Mid-Wicket (Wide)": (0.15, 0.7),
    "Mid-Wicket (Caught)": (0.2, 0.6),
    "Mid Wicket": (0.2, 0.6),
    "Deep Mid-Wicket": (0.1, 0.8),
    "Deep Mid-Wicket (W)": (0.1, 0.8),
    "Square Leg": (0.1, 0.5),
    "Deep Squaring-Leg": (0.05, 0.7),
    "Long Leg": (0.1, 0.2),
    "Fine Leg": (0.15, 0.1),
    "Fine Leg (RHW)": (0.15, 0.1),
    "Fine Leg Covers": (0.2, 0.15),
    "Deep Leg": (0.05, 0.3),
    "Deep Leg Cut": (0.1, 0.4),
    "Deep Leg Cut (W)": (0.1, 0.4),
    "Deep Short Mid Off": (0.3, 0.7),
    
    # Off side positions
    "Long Off": (0.6, 0.9),
    "Long Off (LHW)": (0.6, 0.9),
    "Mid-Off": (0.7, 0.3),
    "Mid Off": (0.7, 0.3),
    "Mid-Off (Wide)": (0.75, 0.35),
    "Deep Mid Off": (0.8, 0.7),
    "Cover": (0.8, 0.4),
    "Covers": (0.8, 0.4),
    "Extra Covers": (0.85, 0.45),
    "Deep Covers": (0.9, 0.6),
    "Deep Cover Cover": (0.9, 0.6),
    "Point": (0.9, 0.3),
    "Forward Point": (0.85, 0.35),
    "Deep Forward Point": (0.95, 0.5),
    
    # Backward positions
    "Deep Backward Spin": (0.9, 0.2),
    
    # Scoop and unusual shots
    "Deceptive Scoop": (0.3, 0.9),
    
    # Pushes and defensive shots
    "Forward Push": (0.5, 0.4),
    "Defensive Push": (0.5, 0.4),
    
    # Dismissal types (center field for visualization)
    "Stumps (Bowled)": (0.5, 0.5),
    "Stumps (LBW)": (0.5, 0.5),
    "Stumps (Caught)": (0.5, 0.5),
    "Bowled Out/Loss Off": (0.5, 0.5),
    
    # Catches
    "Slicing Catch": (0.7, 0.6),
    "Slicing Catch (Long O": (0.6, 0.8),
    
    # Country (assuming this is a regional term for a specific field position)
    "Country": (0.3, 0.8),
    "Country (Wide)": (0.25, 0.85),
}

# 3. Assign coordinates
df1["coords"] = df1["shot_direction"].map(coord_map)
# Use apply(pd.Series) to safely expand coords into x and y columns
coords_df = df1["coords"].apply(pd.Series)
coords_df.columns = ["x", "y"]
df1 = df1.join(coords_df)

# 4. Aggregate counts or total runs
agg = df1.groupby(["x", "y"])["runs"].agg("sum").reset_index()  # use "count" for shot frequency

# 5. Pivot to matrix form
heatmap_data = agg.pivot(index="y", columns="x", values="runs").fillna(0)

# Replace your current image loading section with this:
st.subheader("Field-Position Heatmap of Runs")
st.write("This heatmap uses past game data to show where runs are most often scored, and understanding these hotspots helps predict where the ball will go, allowing for smart ad placement to get the most views.")
try:
    # Try to load the cricket field schematic
    field_img = Image.open("cricket_field_schematic.png")
    
    # Plot heatmap over schematic
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(field_img, extent=(0, 1, 0, 1), aspect="auto")
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="Reds",
        alpha=0.6,
        cbar_kws={"label": "Total Runs"},
        xticklabels=False,
        yticklabels=False
    )
    ax.set_title("Field-Position Heatmap of Runs")  
    ax.axis("off")
    st.pyplot(fig)
    
except FileNotFoundError:
    st.warning("")
    
    # Create heatmap without background image
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="Reds",
        cbar_kws={"label": "Total Runs"},
        annot=True,
        fmt='.0f',
        square=True
    )
    ax.set_title("Field-Position Heatmap of Runs")
    ax.set_xlabel("Field Position (X)")
    ax.set_ylabel("Field Position (Y)")
    st.pyplot(fig)

st.header("Physical And Digital Benchmarks: ")
st.subheader("Digital Benchmarks: ")
st.dataframe(df2, use_container_width=True)
df2["tweet_created_at"] = pd.to_datetime(df2["tweet_created_at"])
df2["time_rounded"] = df2["tweet_created_at"].dt.floor("T")  

view_over_time = df2.groupby("time_rounded")["tweet_view_count"].sum().reset_index()
st.markdown("5. Tweet Engagement: Views Over Time")
st.write("This chart tracks total tweet views over time, which can be used to set a standard for what constitutes viral or highly engaging content for a brand's social media presence")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(view_over_time["time_rounded"], view_over_time["tweet_view_count"], color="purple", marker="o")
ax.set_xlabel("Time")
ax.set_ylabel("Total Views")
ax.set_title("Tweet Views Over Time")
ax.grid(True)
st.pyplot(fig)

st.markdown("6. Tweet Favorites (Likes) Analysis")

try:
    # Total likes
    total_likes = df2["tweet_favorite_count"].sum()

    # Average likes per tweet
    avg_likes = df2["tweet_favorite_count"].mean()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Likes on Tweets", f"{total_likes:,}")
    with col2:
        st.metric("Average Likes per Tweet", f"{avg_likes:.2f}")

    # Optional: Histogram of likes per tweet
    st.subheader("Distribution of Likes per Tweet")
    st.write("This chart provides quantifiable metrics like total and average likes per tweet, establishing a baseline to benchmark current social media performance against set goals or industry averages, and aiding in content strategy improvements.")
    fig_like, ax_like = plt.subplots()
    sns.histplot(df2["tweet_favorite_count"], bins=30, kde=True, color="orange", ax=ax_like)
    ax_like.set_xlabel("Likes per Tweet")
    ax_like.set_ylabel("Tweet Count")
    st.pyplot(fig_like)

except Exception as e:
    st.error(f"Error analyzing likes: {e}")


st.markdown("## 7. Retweet Analysis")

try:
    st.subheader("Retweet Trend Over Time")
    st.write("This chart quantifies content sharing over time, establishing a clear measure of digital reach and potential for virality, which enables brands to set performance standards and compare their engagement against established benchmarks ")
    df2["tweet_created_at"] = pd.to_datetime(df2["tweet_created_at"])
    retweet_trend = df2.groupby(df2["tweet_created_at"].dt.date)["tweet_retweet_count"].sum()
    st.line_chart(retweet_trend)

    st.subheader("Top 5 Most Retweeted Tweets (Pie Chart)")
    st.write("This chart identifies the most widely shared content, establishing a clear measure of digital reach and potential for virality, which enables brands to set performance standards and compare their engagement against established benchmarks.")
    top_retweets = df2.nlargest(5, "tweet_retweet_count")[["tweet_text", "tweet_retweet_count"]]
    fig_ret, ax_ret = plt.subplots()
    ax_ret.pie(top_retweets["tweet_retweet_count"], labels=top_retweets["tweet_text"].str[:40] + "...", 
               autopct="%1.1f%%", startangle=140)
    ax_ret.axis("equal")
    st.pyplot(fig_ret)

except Exception as e:
    st.error(f"Error in retweet analysis: {e}")


st.markdown("## 9. Tweet Text Insights")

try:
    st.subheader("Word Cloud of All Tweets")
    st.write("This word cloud visually identifies the most prominent topics and keywords discussed, establishing a clear standard for the main themes that generate social media engagement, which allows brands to benchmark the relevance and resonance of their messaging and optimize future content strategies.")
    text_data = " ".join(df2["tweet_text"].astype(str).values)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wordcloud, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    st.subheader("Sentiment Analysis")
    st.write("This chart shows what people think and feel about something online. It helps brands understand if their digital plans are working to create a good image. They can also use it to compare their results with other companies or their own goals.")
    df2["sentiment"] = df2["tweet_text"].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    fig_sent, ax_sent = plt.subplots()
    sns.histplot(df2["sentiment"], bins=30, kde=True, ax=ax_sent, color="purple")
    ax_sent.set_xlabel("Sentiment Score (-1 = Negative, +1 = Positive)")
    st.pyplot(fig_sent)

    st.subheader("Top 5 Viral Phrases (by Likes + Retweets)")
    st.write("This chart shows which posts got the most likes and shares. It helps set a clear example of what kind of content goes viral, so brands can use it to improve future posts and plan better strategies.")
    df2["virality"] = df2["tweet_favorite_count"] + df2["tweet_retweet_count"]
    top_viral = df2.nlargest(5, "virality")[["tweet_text", "virality"]]
    st.table(top_viral)

except Exception as e:
    st.error(f"Text analysis error: {e}")


st.subheader("Physical benchmarks:")
st.write("This table shows clear numbers about stadium sizes, like how long the boundaries are. It helps brands understand the physical space so they can compare stadiums and plan where to put their ads to get the most attention.")
st.dataframe(df3, use_container_width=True)
  # Your wagon wheel
image_path = "Screenshot 2025-06-19 090336.png"
image_path1 = "Screenshot 2025-06-19 090810.png"
image_path2 = "Screenshot 2025-06-19 091356.png"
image_path3 = "Screenshot 2025-06-19 091405.png"
image = Image.open(image_path)
image1 = Image.open(image_path1)
image2= Image.open(image_path2)
image3= Image.open(image_path3)
# Layout: 1 column for wagon wheel, 1 column for the two right images
col1, col2 = st.columns([2, 1])  # adjust width ratio as needed

# Left: Wagon Wheel
st.image(image, caption="Wagon Wheel Shot Map: Narendar Modi Stadium", use_container_width=True)
st.image(image1, use_container_width=True)

# Right: Two stacked images

st.image(image2, caption="Wagon Wheel Shot Map: M. Chinnaswamy Stadium", use_container_width=True)
st.image(image3, use_container_width=True)
