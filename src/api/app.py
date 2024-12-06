from flask import Flask, request, jsonify
from tasks import generate_predictions_task  # Import the Celery task
from celery import Celery

# Initialize Flask app
app = Flask(__name__)

# Configure Celery
app.config["CELERY_BROKER_URL"] = "redis://localhost:6379/0"
app.config["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"


def make_celery(app):
    """Initialize the Celery instance."""
    celery = Celery(
        app.import_name,
        backend=app.config["CELERY_RESULT_BACKEND"],
        broker=app.config["CELERY_BROKER_URL"],
    )
    celery.conf.update(app.config)
    return celery


celery = make_celery(app)


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to accept requests and return a task token."""
    try:
        # Get the input data from the request
        data = request.json
        input_text = data.get("input_text", "")

        # Start Celery task
        task = generate_predictions_task.apply_async(args=[input_text])

        # Return the task ID as the response token
        return jsonify({"task_id": task.id}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/results/<task_id>", methods=["GET"])
def get_results(task_id):
    """Endpoint to retrieve results for a specific task."""
    # Check the task status
    task_result = generate_predictions_task.AsyncResult(task_id)

    if task_result.state == "SUCCESS":
        return jsonify({"status": "completed", "predictions": task_result.result})
    elif task_result.state == "FAILURE":
        return jsonify({"status": "failed", "error": str(task_result.info)})
    else:
        return jsonify({"status": task_result.state.lower()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
