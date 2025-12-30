from flask import Flask
import redis

app = Flask(__name__)

# Connect to Redis container (hostname matches service name in podman-compose.yaml)
r = redis.Redis(host="redis", port=6379, decode_responses=True)


@app.route("/")
def hello():
    """Main page that tracks and displays visit count."""
    try:
        count = r.incr("hits")
        return f"Hello! This page has been visited {count} times.\n"
    except redis.ConnectionError:
        return "Error: Could not connect to Redis database.\n", 500


if __name__ == "__main__":
    # Run on all interfaces so it's accessible from outside the container
    app.run(host="0.0.0.0", port=5000, debug=False)
