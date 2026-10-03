const API_URL =
  import.meta.env.VITE_API_URL?.trim() ||
  "/api/ask";

export const apiConfigured = Boolean(API_URL);

export async function askReceipt(file, question) {
  if (!file) {
    throw new Error("No receipt image was provided.");
  }

  if (!question?.trim()) {
    throw new Error("Please enter a question.");
  }

  const formData = new FormData();

  formData.append("image", file);
  formData.append("question", question.trim());

  const response = await fetch(API_URL, {
    method: "POST",
    body: formData,
  });

  let payload;

  try {
    payload = await response.json();
  } catch {
    throw new Error(
      `Inference server returned HTTP ${response.status}.`
    );
  }

  if (!response.ok) {
    throw new Error(
      payload?.error ||
        `Inference server returned HTTP ${response.status}.`
    );
  }

  const answer = payload?.answer;

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