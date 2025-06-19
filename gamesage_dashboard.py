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


       


    # Section 1: Sponsor Detection Overview
    st.markdown("## 1. Sponsor Detection Overview")
    st.write("This section shows how many frames had sponsor names and how many did not.")

    # Count total frames (rows in audio_df)
    total_frames = len(audio_df)
    missed_frames = (audio_df["VisibleSponsorsDuringPeak"].str.strip() == "NoSponsorDetected").sum()
    detected_frames = total_frames - missed_frames

    detection_data = pd.DataFrame({
        "Detection Status": ["Detected", "Missed"],
        "Frame Count": [detected_frames, missed_frames]
    })

    st.bar_chart(detection_data.set_index("Detection Status"))


        # Section 4: Match Moments Sponsor Count
    st.markdown("## 4. Match Moments Sponsor Count")
    st.write("This section shows how many sponsors appeared at each significant moment (based on audio peaks).")

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

    st.subheader("Raw Sponsor Data Table")
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
st.header("Map targeting")
components.iframe(
    "https://sponsor-map-33vc.vercel.app",
    height=600,
    width=1000,
    scrolling=True
)
st.header("Stadium Analysis")

st.subheader("Raw Ball-by-Ball Data (Innings 2)")
st.dataframe(df1, use_container_width=True)

# Assume df1 is already loaded via load_data()
# df1 = pd.read_csv("cricket_shots.csv")  # already executed in your load_data()

st.subheader("Shot-Direction Frequency (using df1)")

# 1. Count occurrences of each shot direction
shot_counts = df1['shot_direction'].value_counts().sort_index()  # pandas.Series.value_counts()[1]

# 2. Render a simple bar chart in Streamlit
st.bar_chart(shot_counts)  # Streamlit bar_chart API[2]


shot_counts = df1['shot_direction'].value_counts().sort_index()
shot_avg_runs = df1.groupby('shot_direction')['runs'].mean().loc[shot_counts.index]

# Streamlit visualization
st.subheader("Average Runs by Shot Direction")

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
    st.warning("Cricket field image not found. Showing heatmap without background.")
    
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
st.markdown("## 5. Tweet Engagement: Views Over Time")
st.write("This line chart shows how total tweet views changed over time during the match.")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(view_over_time["time_rounded"], view_over_time["tweet_view_count"], color="purple", marker="o")
ax.set_xlabel("Time")
ax.set_ylabel("Total Views")
ax.set_title("Tweet Views Over Time")
ax.grid(True)
st.pyplot(fig)

st.markdown("## 6. Tweet Favorites (Likes) Analysis")

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
    df2["tweet_created_at"] = pd.to_datetime(df2["tweet_created_at"])
    retweet_trend = df2.groupby(df2["tweet_created_at"].dt.date)["tweet_retweet_count"].sum()
    st.line_chart(retweet_trend)

    st.subheader("Top 5 Most Retweeted Tweets (Pie Chart)")
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
    text_data = " ".join(df2["tweet_text"].astype(str).values)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wordcloud, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    st.subheader("Sentiment Analysis")
    df2["sentiment"] = df2["tweet_text"].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    fig_sent, ax_sent = plt.subplots()
    sns.histplot(df2["sentiment"], bins=30, kde=True, ax=ax_sent, color="purple")
    ax_sent.set_xlabel("Sentiment Score (-1 = Negative, +1 = Positive)")
    st.pyplot(fig_sent)

    st.subheader("Top 5 Viral Phrases (by Likes + Retweets)")
    df2["virality"] = df2["tweet_favorite_count"] + df2["tweet_retweet_count"]
    top_viral = df2.nlargest(5, "virality")[["tweet_text", "virality"]]
    st.table(top_viral)

except Exception as e:
    st.error(f"Text analysis error: {e}")


st.subheader("Physical benchmarks:")
st.dataframe(df3, use_container_width=True)


st.title('Cricket Wagon Wheel - Narendra Modi Stadium Ahmedabad')
#st.set_option('deprecation.showPyplotGlobalUse', False)

# Create a figure and axis with green background representing the cricket ground
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('green')

# Draw the circular boundary of the cricket ground
circle = plt.Circle((0, 0), 1, color='white', fill=False, linewidth=2)
ax.add_artist(circle)

# Define the zones around the ground
zones = ['Third Man', 'Point', 'Cover', 'Long Off', 'Long On', 'Mid Wicket', 'Square Leg', 'Fine Leg']
angles = np.linspace(0, 2 * np.pi, len(zones), endpoint=False)

# Plot zone labels around the circle
for angle, zone in zip(angles, zones):
    x = 1.1 * np.cos(angle)
    y = 1.1 * np.sin(angle)
    ax.text(x, y, zone, color='white', fontsize=10, ha='center', va='center', fontweight='bold')

# Mark the distance measurements on the ground
# Straight (end-to-end) 75 m-89 m
# Square (off-side & leg-side) 66 m-74 m
# Behind the wicket 62 m-71 m

# Convert distances to relative scale (max radius = 1)
max_straight = 89
max_square = 74
max_behind = 71

# Draw concentric arcs for each distance range
# Straight line (end-to-end) - vertical line
ax.plot([0, 0], [-75/max_straight, 75/max_straight], color='yellow', linewidth=2, label='Straight 75-89 m')
ax.plot([0, 0], [-89/max_straight, 89/max_straight], color='yellow', linewidth=1, linestyle='dashed')

# Square leg and off side arcs
square_angles = np.linspace(np.pi/2, 3*np.pi/2, 100)
ax.plot(66/max_square * np.cos(square_angles), 66/max_square * np.sin(square_angles), color='orange', linewidth=2, label='Square 66-74 m')
ax.plot(74/max_square * np.cos(square_angles), 74/max_square * np.sin(square_angles), color='orange', linewidth=1, linestyle='dashed')

# Behind the wicket arcs (bottom half)
behind_angles = np.linspace(np.pi, 2*np.pi, 100)
ax.plot(62/max_behind * np.cos(behind_angles), 62/max_behind * np.sin(behind_angles), color='red', linewidth=2, label='Behind 62-71 m')
ax.plot(71/max_behind * np.cos(behind_angles), 71/max_behind * np.sin(behind_angles), color='red', linewidth=1, linestyle='dashed')

# Set limits and aspect
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.axis('off')

# Show legend
ax.legend(loc='upper right', fontsize=8)

# Show the plot in Streamlit
st.pyplot(fig)
