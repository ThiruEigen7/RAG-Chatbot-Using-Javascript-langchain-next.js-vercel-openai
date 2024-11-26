import { DataAPIClient } from "@datastax/astra-db-ts";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import axios from "axios";
import "dotenv/config";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// Load environment variables
const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  NOMIC_API_TOKEN,
} = process.env;

// Data for scraping and embeddings
const fin_data = [
  "https://en.wikipedia.org/wiki/Stock",
  "https://en.wikipedia.org/wiki/Mutual_fund",
  "https://en.wikipedia.org/wiki/Financial_statement",
  "https://www.financialexpress.com/",
  "https://www.business-standard.com/finance",
];

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT, { namespace: ASTRA_DB_NAMESPACE });

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 100,
});

// Function to create a collection with vector embeddings
const createCollection = async () => {
  const res = await db.createCollection(ASTRA_DB_COLLECTION, {
    vector: {
      dimension: 768, // Nomic embedding dimension
      metric: "dot_product",
    },
  });
  console.log(res);
};

// Function to scrape web pages
const scrapePage = async (url) => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: true,
    },
    gotoOptions: {
      waitUntil: "domcontentloaded",
    },
    evaluate: async (page, browser) => {
      const result = await page.evaluate(() => document.body.innerHTML);
      await browser.close();
      return result;
    },
  });
  return (await loader.scrape())?.replace(/<[^>]*>?/gm, ""); // Remove HTML tags
};

// Function to generate embeddings using Nomic API
const getNomicEmbeddings = async (texts) => {
  try {
    const data = JSON.stringify({
      texts,
      task_type: "search_document",
      max_tokens_per_text: 8192,
      dimensionality: 768,
    });

    const config = {
      method: "post",
      maxBodyLength: Infinity,
      url: "https://api-atlas.nomic.ai/v1/embedding/text",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        Authorization: `Bearer ${NOMIC_API_TOKEN}`,
      },
      data,
    };

    const response = await axios.request(config);
    return response.data.embeddings; // Return the embedding array
  } catch (error) {
    console.error("Error generating embeddings:", error.response?.data || error.message);
    throw error;
  }
};

// Function to load sample data into the database
const loadSampleData = async () => {
  const collection = await db.collection(ASTRA_DB_COLLECTION);
  for await (const url of fin_data) {
    const content = await scrapePage(url);
    const chunks = await splitter.splitText(content);
    for await (const chunk of chunks) {
      try {
        const [vector] = await getNomicEmbeddings([chunk]); // Nomic API embedding call
        const res = await collection.insertOne({
          $vector: vector,
          text: chunk,
        });
        console.log("Inserted:", res);
      } catch (error) {
        console.error("Error inserting chunk:", error.message);
      }
    }
  }
};

// Main execution flow
(async () => {
  try {
    await createCollection();
    await loadSampleData();
    console.log("Data loaded successfully!");
  } catch (error) {
    console.error("Error during execution:", error.message);
  }
})();
