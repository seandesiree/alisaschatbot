from youtube_transcript_api import YouTubeTranscriptApi
import re
import os

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com/watch\?v=)([^&\n?#]+)',
        r'(?:youtu\.be/)([^&\n?#]+)',
        r'(?:youtube\.com/embed/)([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(url, title="Interview"):
    """Extract transcript from YouTube URL"""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            print(f"Could not extract video ID from {url}")
            return None
            
        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine all text with timestamps for context
        full_transcript = f"YouTube Interview: {title}\n\n"
        for item in transcript_list:
            full_transcript += f"[{int(item['start']//60):02d}:{int(item['start']%60):02d}] {item['text']}\n"
        
        return full_transcript
        
    except Exception as e:
        print(f"Error getting transcript for {url}: {e}")
        return None

# Test your URLs
if __name__ == "__main__":
    urls = [
        # Add your YouTube URLs here
        "https://www.youtube.com/watch?v=nhZyoxUTOYo",
        "https://www.youtube.com/watch?v=6INOECGMhIo",
        "https://www.youtube.com/watch?v=PerR6YDhTo0",
        "https://www.youtube.com/watch?v=djCayGzqg1M",
        "https://www.youtube.com/watch?v=TsNsryumZdY",
        "https://www.youtube.com/watch?v=xRsFKECfLHY",
        "https://www.youtube.com/watch?v=bP1nPkKfyWc",
        
    ]
    
    for i, url in enumerate(urls):
        transcript = get_youtube_transcript(url, f"Interview_{i+1}")
        if transcript:
            with open(f"artist_data/interview_{i+1}.txt", "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"✅ Saved interview_{i+1}.txt")
        else:
            print(f"❌ Failed to get transcript from {url}")