const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const metadataPath = path.join("runner-temp", "netlify-metadata", "metadata.json");
const sourceSha = "637a37bb57e1a9993e28d320804aaed7bb3fe56f";

function loadScript(filename) {
  const workflow = fs
    .readFileSync(path.join(__dirname, "../workflows", filename), "utf8")
    .replace(/\r\n/g, "\n");
  const script = workflow
    .split("          script: |\n")[1]
    .split("\n      - name:")[0]
    .replace(/^            /gm, "");
  return new AsyncFunction("github", "context", "core", "require", "process", script);
}

const resolvePullRequest = loadScript("deploy-netlify-preview.yml");
const writeMetadata = loadScript("quarto-ci.yml");

async function resolve(options = {}) {
  const workflowRun = {
    id: 123,
    head_sha: sourceSha,
    head_branch: "docs-update",
    head_repository: { full_name: "contributor/fork" },
    pull_requests: [],
    ...options.run,
  };
  const pullRequest = {
    number: 49,
    state: "open",
    base: { repo: { full_name: "owner/repo" } },
    head: {
      sha: workflowRun.head_sha,
      ref: workflowRun.head_branch,
      repo: { full_name: workflowRun.head_repository.full_name },
    },
    ...options.pr,
  };
  const metadata = Object.hasOwn(options, "metadata")
    ? options.metadata
    : { pr_number: 49, head_sha: sourceSha };
  const outputs = {};
  const github = {
    rest: {
      pulls: {
        get: async (params) => {
          assert.deepEqual(params, { owner: "owner", repo: "repo", pull_number: 49 });
          return { data: pullRequest };
        },
      },
    },
  };
  await resolvePullRequest(
    github,
    {
      repo: { owner: "owner", repo: "repo" },
      payload: { workflow_run: workflowRun, repository: { full_name: "owner/repo" } },
    },
    {
      setOutput: (name, value) => { outputs[name] = value; },
      info: () => {},
    },
    (name) => {
      assert.equal(name, "node:fs");
      return {
        readFileSync: (filename, encoding) => {
          assert.equal(filename, metadataPath);
          assert.equal(encoding, "utf8");
          return options.rawJson ?? JSON.stringify(metadata);
        },
      };
    },
    { env: { METADATA_PATH: metadataPath } },
  );
  return outputs;
}

test("passes PR metadata from the render workflow to a fork preview", async () => {
  let json;
  await writeMetadata(
    {},
    {
      sha: "synthetic-merge-sha",
      payload: { pull_request: { number: 49, head: { sha: sourceSha } } },
    },
    {},
    (name) => name === "node:path" ? path : {
      mkdirSync: (directory, options) => {
        assert.equal(directory, path.dirname(metadataPath));
        assert.deepEqual(options, { recursive: true });
      },
      writeFileSync: (filename, content) => {
        assert.equal(filename, metadataPath);
        json = content;
      },
    },
    { env: { METADATA_PATH: metadataPath } },
  );
  assert.deepEqual(JSON.parse(json), { pr_number: 49, head_sha: sourceSha });
  assert.deepEqual(await resolve({ rawJson: json }), { number: 49, head_sha: sourceSha });
});

test("uses JSON even when event PR metadata is present", async () => {
  assert.deepEqual(
    await resolve({ run: { pull_requests: [{ number: 48 }] } }),
    { number: 49, head_sha: sourceSha },
  );
});

for (const metadata of [
  null, {}, { pr_number: "49" }, { pr_number: 0 },
  { pr_number: 1.5 }, { pr_number: Number.MAX_SAFE_INTEGER + 1 },
]) {
  test(`rejects invalid PR metadata: ${JSON.stringify(metadata)}`, async () => {
    await assert.rejects(resolve({ metadata }), /Invalid PR number/);
  });
}

test("rejects malformed JSON", async () => {
  await assert.rejects(resolve({ rawJson: "{" }), SyntaxError);
});

for (const head_sha of [undefined, 123, "", "another-run-sha"]) {
  test(`rejects a missing or mismatched metadata SHA: ${head_sha}`, async () => {
    await assert.rejects(
      resolve({ metadata: { pr_number: 49, head_sha } }),
      /Metadata commit SHA does not match/,
    );
  });
}

for (const head of [
  { repo: { full_name: "unrelated/fork" }, ref: "docs-update" },
  { repo: { full_name: "contributor/fork" }, ref: "other-branch" },
  { repo: null, ref: "docs-update" },
]) {
  test(`rejects a mismatched PR source: ${JSON.stringify(head)}`, async () => {
    await assert.rejects(resolve({ pr: { head } }), /does not match/);
  });
}

test("rejects a PR targeting another repository", async () => {
  await assert.rejects(
    resolve({ pr: { base: { repo: { full_name: "other/repo" } } } }),
    /does not match/,
  );
});

test("skips a PR updated since the render started", async () => {
  assert.deepEqual(await resolve({ pr: { head: {
    sha: "newer-sha", ref: "docs-update", repo: { full_name: "contributor/fork" },
  } } }), {});
});

test("skips a closed PR", async () => {
  assert.deepEqual(await resolve({ pr: { state: "closed" } }), {});
});
