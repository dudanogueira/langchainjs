/* eslint-disable no-process-env */
import { test, expect } from "@jest/globals";
import weaviate, { Filters, WeaviateClient } from "weaviate-client";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { WeaviateStore } from "@langchain/weaviate";

// get one client for all tests
async function getWeaviateClient() {
  if (process.env.WEAVIATE_CLOUD_URL && process.env.WEAVIATE_CLOUD_API_KEY) {
    const client: WeaviateClient = await weaviate.connectToWeaviateCloud(
      process.env.WEAVIATE_CLOUD_URL, // Replace with your Weaviate Cloud URL
      {
        authCredentials: new weaviate.ApiKey(process.env.WEAVIATE_CLOUD_API_KEY), // Replace with your Weaviate Cloud API key
      }
    );
    return client;
  } else {
    const client = await weaviate.connectToCustom(
    {
        httpHost: process.env.WEAVIATE_HTTP_URL || "localhost",  // URL only, no http prefix
        httpPort: process.env.WEAVIATE_HTTP_PORT ? parseInt(process.env.WEAVIATE_HTTP_PORT, 10) : 8080,
        httpSecure: process.env.WEAVIATE_HTTP_SECURE ? process.env.WEAVIATE_HTTP_SECURE === "true" : false, 
        grpcHost: process.env.WEAVIATE_GPC_URL || "localhost", // URL only, no http prefix
        grpcPort: process.env.WEAVIATE_GPC_PORT ? parseInt(process.env.WEAVIATE_GPC_PORT, 10) : 50051, 
        grpcSecure: process.env.WEAVIATE_GPC_SECURE ? process.env.WEAVIATE_GPC_SECURE === "true" : false, 
        authCredentials: process.env.WEAVIATE_API_KEY ? new weaviate.ApiKey(process.env.WEAVIATE_API_KEY): undefined,
      })
      return client
  }
}

type DocumentWithoutId = Omit<Document, 'id'>;
async function removeIds(results: Document[]): Promise<DocumentWithoutId[]> {
  return results.map(({ id, ...rest }) => rest);
}

test("WeaviateStore", async () => {
  const client = await getWeaviateClient();
  await client.collections.delete("Test");
  const weaviateArgs = {
    client,
    indexName: "Test",
    textKey: "text",
    metadataKeys: ["foo"],
  };
  const store = await WeaviateStore.fromTexts(
    ["hello world", "hi there", "how are you", "bye now"],
    [{ foo: "bar" }, { foo: "baz" }, { foo: "qux" }, { foo: "bar" }],
    new OpenAIEmbeddings(),
    weaviateArgs
  );
  const collection = client.collections.get(weaviateArgs.indexName);
  const results = await store.similaritySearch("hello world", 1);
  
  expect(await removeIds(results)).toEqual([
    new Document({ pageContent: "hello world", metadata: { foo: "bar" } }),
  ]);

  // TEST FILTERING
  const results2 = await store.similaritySearch(
    "hello world",
    1,
    Filters.and(collection.filter.byProperty("foo").equal("baz"))
  );

  expect(await removeIds(results2)).toEqual([
    new Document({ pageContent: "hi there", metadata: { foo: "baz" } }),
  ]);

  const testDocumentWithObjectMetadata = new Document({
    pageContent: "this is the deep document world!",
    metadata: {
      deep: {
        string: "deep string",
        deepdeep: {
          string: "even a deeper string",
        },
      },
    },
  });
  await client.collections.delete("DocumentTest");
  const documentStore = await WeaviateStore.fromDocuments(
    [testDocumentWithObjectMetadata],
    new OpenAIEmbeddings(),
    {
      client,
      indexName: "DocumentTest",
      textKey: "text",
      metadataKeys: ["deep_string", "deep_deepdeep_string"],
    }
  );
  const result3 = await documentStore.similaritySearch(
    "this is the deep document world!",
    1,
    Filters.and(
      collection.filter.byProperty("deep_string").equal("deep string")
    )
  );
  
  expect(await removeIds(result3)).toEqual([
    new Document({
      pageContent: "this is the deep document world!",
      metadata: {
        deep_string: "deep string",
        deep_deepdeep_string: "even a deeper string",
      },
    }),
  ]);
});

test("WeaviateStore returning only MetadataKeys", async () => {
  const client = await getWeaviateClient();
  await client.collections.delete("TestMetadataKeys");
  const weaviateArgs = {
    client,
    indexName: "TestMetadataKeys",
    textKey: "text",
    metadataKeys: ["foo", ],
  };
  const store = await WeaviateStore.fromTexts(
    ["hello world", ],
    [{ foo: "bar", bar: "foo" }],
    new OpenAIEmbeddings(),
    weaviateArgs
  );
  const results = await store.similaritySearch("hello world", 1);

  expect(await removeIds(results)).toEqual([
    new Document({ pageContent: "hello world", metadata: { foo: "bar" } }),
  ]);
});

test("WeaviateStore upsert + delete", async () => {

  const client = await getWeaviateClient();
  const createdAt = new Date().getTime();
  await client.collections.delete("DocumentTest");
  const weaviateArgs = {
    client,
    indexName: "DocumentTest",
    textKey: "pageContent",
    metadataKeys: ["deletionTest"],
  };
  const store = await WeaviateStore.fromDocuments(
    [
      new Document({
        pageContent: "testing",
        metadata: { deletionTest: createdAt.toString() },
      }),
    ],
    new OpenAIEmbeddings(),
    weaviateArgs
  );
  const collection = client.collections.get(weaviateArgs.indexName);
  const ids = await store.addDocuments([
    {
      pageContent: "hello world",
      metadata: { deletionTest: (createdAt + 1).toString() },
    },
    {
      pageContent: "hello world",
      metadata: { deletionTest: (createdAt + 1).toString() },
    },
  ]);
  const results = await store.similaritySearch(
    "hello world",
    4,
    Filters.and(
      collection.filter
        .byProperty("deletionTest")
        .equal((createdAt + 1).toString())
    )
  );
  expect(await removeIds(results)).toEqual([
    new Document({
      pageContent: "hello world",
      metadata: { deletionTest: (createdAt + 1).toString() },
    }),
    new Document({
      pageContent: "hello world",
      metadata: { deletionTest: (createdAt + 1).toString() },
    }),
  ]);

  const ids2 = await store.addDocuments(
    [
      {
        pageContent: "hello world upserted",
        metadata: { deletionTest: (createdAt + 1).toString() },
      },
      {
        pageContent: "hello world upserted",
        metadata: { deletionTest: (createdAt + 1).toString() },
      },
    ],
    { ids }
  );

  expect(ids2).toEqual(ids);
  const results2 = await store.similaritySearch(
    "hello world",
    4,
    Filters.and(
      collection.filter
        .byProperty("deletionTest")
        .equal((createdAt + 1).toString())
    )
  );
  expect(await removeIds(results2)).toEqual([
    new Document({
      pageContent: "hello world upserted",
      metadata: { deletionTest: (createdAt + 1).toString() },
    }),
    new Document({
      pageContent: "hello world upserted",
      metadata: { deletionTest: (createdAt + 1).toString() },
    }),
  ]);

  await store.delete({ ids: ids.slice(0, 1) });

  const results3 = await store.similaritySearch(
    "hello world",
    1,
    Filters.and(
      collection.filter
        .byProperty("deletionTest")
        .equal((createdAt + 1).toString())
    )
  );
  expect(await removeIds(results3)).toEqual([
    new Document({
      pageContent: "hello world upserted",
      metadata: { deletionTest: (createdAt + 1).toString() },
    }),
  ]);
});

test("WeaviateStore delete with filter", async () => {

  const client = await getWeaviateClient();
  await client.collections.delete("FilterDeletionTest");
  const weaviateArgs = {
    client,
    indexName: "FilterDeletionTest",
    textKey: "text",
    metadataKeys: ["foo"],
  };
  const store = await WeaviateStore.fromTexts(
    ["hello world", "hi there", "how are you", "bye now"],
    [{ foo: "bar" }, { foo: "baz" }, { foo: "qux" }, { foo: "bar" }],
    new OpenAIEmbeddings(),
    weaviateArgs
  );
  const collection = client.collections.get(weaviateArgs.indexName);
  const results = await store.similaritySearch("hello world", 1);
  expect(await removeIds(results)).toEqual([
    new Document({ pageContent: "hello world", metadata: { foo: "bar" } }),
  ]);
  await store.delete({
    filter: collection.filter.byProperty("foo").equal("bar"),
  });
  await store.delete({
    filter: collection.filter.byProperty("foo").equal("bar"),
  });
  const results2 = await store.similaritySearch(
    "hello world",
    1,
    collection.filter.byProperty("foo").equal("bar")
  );
  expect(results2).toEqual([]);
});
