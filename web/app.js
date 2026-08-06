const output = document.getElementById("output");

function splitList(value) {
  if (!value || !value.trim()) {
    return [];
  }
  return value
    .split(/[,|]/)
    .map((v) => v.trim())
    .filter(Boolean);
}

function setOutput(data) {
  output.textContent =
    typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

async function callApi(path, method, body) {
  try {
    const res = await fetch(path, {
      method,
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: body ? JSON.stringify(body) : undefined,
    });
    const text = await res.text();
    let payload = text;
    try {
      payload = JSON.parse(text);
    } catch (_err) {
      // Keep plain text payload.
    }
    if (!res.ok) {
      setOutput({ status: res.status, error: payload });
      return;
    }
    setOutput(payload);
  } catch (err) {
    setOutput(`Request failed: ${err}`);
  }
}

document.getElementById("screenBtn").addEventListener("click", async () => {
  const topKRaw = document.getElementById("topk").value;
  const body = {
    name: document.getElementById("name").value,
    first_name: document.getElementById("firstName").value,
    middle_name: document.getElementById("middleName").value,
    last_name: document.getElementById("lastName").value,
    dob: document.getElementById("dob").value,
    residency: document.getElementById("residency").value,
    nationality: document.getElementById("nationality").value,
    aliases: splitList(document.getElementById("aliases").value),
    relative_names: splitList(document.getElementById("relativeNames").value),
    gender: document.getElementById("gender").value,
    top_k: Number(topKRaw || 3),
  };
  await callApi("/screen", "POST", body);
});

document.getElementById("batchBtn").addEventListener("click", async () => {
  try {
    const body = JSON.parse(document.getElementById("batchPayload").value);
    await callApi("/screen/batch", "POST", body);
  } catch (err) {
    setOutput(`Invalid JSON batch payload: ${err}`);
  }
});

document.getElementById("healthBtn").addEventListener("click", async () => {
  await callApi("/health", "GET");
});

document.getElementById("rebuildBtn").addEventListener("click", async () => {
  await callApi("/index/rebuild", "POST");
});
