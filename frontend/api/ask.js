import { Client, handle_file } from "@gradio/client";

const HF_SPACE = "shekharrrr/receiptqa-demo";

let clientPromise;

function getClient() {
  const token = process.env.HF_TOKEN;

  if (!token) {
    throw new Error("HF_TOKEN is not configured on Vercel.");
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

export const maxDuration = 60;

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

export async function GET() {
  return Response.json(
    {
      ok: true,
      message: "ReceiptQA API is running.",
    },
    {
      status: 200,
      headers: corsHeaders,
    }
  );
}

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: corsHeaders,
  });
}

export async function POST(request) {
  try {
    const formData = await request.formData();

    const image = formData.get("image");
    const question = formData.get("question");

    if (!(image instanceof File)) {
      return Response.json(
        {
          error: "Please upload a receipt image.",
        },
        {
          status: 400,
          headers: corsHeaders,
        }
      );
    }

    if (
      typeof question !== "string" ||
      !question.trim()
    ) {
      return Response.json(
        {
          error: "Please enter a question.",
        },
        {
          status: 400,
          headers: corsHeaders,
        }
      );
    }

    const client = await getClient();

    const result = await client.predict("/ask", {
      image: handle_file(image),
      question: question.trim(),
    });

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
        headers: corsHeaders,
      }
    );
  } catch (error) {
    console.error(
      "ReceiptQA inference error:",
      error
    );

    return Response.json(
      {
        error:
          error?.message ||
          "Something went wrong while running the model.",
      },
      {
        status: 500,
        headers: corsHeaders,
      }
    );
  }
}