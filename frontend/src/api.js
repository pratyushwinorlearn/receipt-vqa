import { Client, handle_file } from "@gradio/client";

const API_URL =
  import.meta.env.VITE_API_URL?.trim();

const API_NAME =
  import.meta.env.VITE_API_NAME ||
  "/ask";

let clientPromise;

export const apiConfigured =
  Boolean(API_URL);

function getClient() {
  if (!API_URL) {
    throw new Error(
      "API URL is not configured. Set VITE_API_URL and rebuild."
    );
  }

  if (!clientPromise) {
    clientPromise =
      Client.connect(API_URL).catch(
        (error) => {
          clientPromise = undefined;
          throw error;
        }
      );
  }

  return clientPromise;
}

export async function askReceipt(
  file,
  question
) {
  if (!file) {
    throw new Error(
      "No receipt image was provided."
    );
  }

  if (!question?.trim()) {
    throw new Error(
      "Please enter a question."
    );
  }

  const client =
    await getClient();

  const result =
    await client.predict(
      API_NAME,
      {
        image: handle_file(file),
        question:
          question.trim(),
      }
    );

  const answer =
    result?.data?.[0];

  if (
    typeof answer !== "string" ||
    !answer.trim()
  ) {
    throw new Error(
      "The model returned an empty answer."
    );
  }

  return answer.trim();
}