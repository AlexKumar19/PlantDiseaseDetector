document.addEventListener("DOMContentLoaded", function() {
    const startButton = document.getElementById("start-button");
    const stopButton = document.getElementById("stop-button");
    const videoFeed = document.getElementById("video-feed");

    startButton.addEventListener("click", function() {
        videoFeed.style.display = "block";
        videoFeed.src = "/video_feed";  // Start the stream
    });

    stopButton.addEventListener("click", function() {
        videoFeed.style.display = "none";
        videoFeed.src = "";  // Stop the stream
    });
});
