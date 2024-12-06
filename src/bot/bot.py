import os
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import requests

# Load the bot token from the environment
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError(
        "Telegram bot token not set in TELEGRAM_BOT_TOKEN environment variable"
    )

# Define the base URL for your Flask API
API_BASE_URL = "http://app:5000"  # Use the container name "app" to communicate within the Docker network


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the bot is started."""
    await update.message.reply_text(
        "Привет! Я обожаю отгадывать загадки. Отправь мне загадку, а я попробую её разгадать!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages."""
    user_input = update.message.text

    try:
        # Send the user input to the Flask API
        response = requests.post(
            f"{API_BASE_URL}/predict", json={"input_text": user_input}
        )
        if response.status_code == 202:
            task_id = response.json()["task_id"]

            # Poll the results endpoint
            while True:
                result_response = requests.get(f"{API_BASE_URL}/results/{task_id}")
                result_data = result_response.json()

                if result_data["status"] == "completed":
                    predictions = result_data["predictions"]
                    top_guess = predictions[0]
                    alternative_guesses = ", ".join(predictions[1:])
                    reply = (
                        f"Скорее всего, это {top_guess}. "
                        f"Но, возможно, и что-то из этого: {alternative_guesses}."
                    )
                    await update.message.reply_text(reply)
                    break
                elif result_data["status"] == "failed":
                    await update.message.reply_text(
                        "Извините, произошла ошибка при обработке запроса."
                    )
                    break
        else:
            await update.message.reply_text("Ошибка: я не смог обработать ваш запрос.")

    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка: {str(e)}")


def main():
    """Start the bot."""
    # Create the application
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    print("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()
