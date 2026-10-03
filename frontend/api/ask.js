import { Client, handle_file } from "@gradio/client";

const HF_SPACE =
  "shekharrrr/receiptqa-demo";

export const maxDuration = 60;

let clientPromise;

function getClient() {
  const token = process.env.HF_TOKEN;

  if (!token) {
    throw new Error(
      "HF_TOKEN is not configured on Vercel."
    );
  }

  if (!clientPromise) {
    clientPromise = Client.connect(HF_SPACE, {
      token,
    }).catch((error) => {
      clientPromise = undefined;
      throw error;
    });
  }

  return clientPromise;
}

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

export default async function handler(request) {
  const headers = corsHeaders();

  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers,
    });
  }

  if (request.method !== "POST") {
    return Response.json(
      {
        error: "Method not allowed.",
      },
      {
        status: 405,
        headers,
      }
    );
  }

  try {
    const formData = await request.formData();

    const image = formData.get("image");
    const question = formData.get("question");

    if (!(image instanceof File)) {
      return Response.json(
        {
          error:
            "Please upload a receipt image.",
        },
        {
          status: 400,
          headers,
        }
      );
    }

    if (
      typeof question !== "string" ||
      !question.trim()
    ) {
      return Response.json(
        {
          error:
            "Please enter a question.",
        },
        {
          status: 400,
          headers,
        }
      );
    }

    const client = await getClient();

    const result = await client.predict(
      "/ask",
      {
        image: handle_file(image),
        question: question.trim(),
      }
    );

    const answer = result?.data?.[0];

    if (
      typeof answer !== "string" ||
      !answer.trim()
    ) {
      throw new Error(
        "The model returned an empty answer."
      );
    }

    return Response.json(
      {
        answer: answer.trim(),
      },
      {
        status: 200,
        headers,
      }
    );
  } catch (error) {
    console.error("ReceiptQA inference error:", error);

    return Response.json(
      {
        error:
          error?.message ||
          "Something went wrong while running the model.",
      },
      {
        status: 500,
        headers,
      }
    );
  }
}