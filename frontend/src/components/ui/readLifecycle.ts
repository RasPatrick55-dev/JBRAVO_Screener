// Slow provider-backed reads observed in production can exceed 24 seconds.
// Keep a finite deadline for both the response and its decoded body.
export const READ_TIMEOUT_MS = 45_000;
export const REFRESH_INTERVAL_MS = 20_000;

export async function deadlineRead<T>(operation: (signal: AbortSignal) => Promise<T>, milliseconds = READ_TIMEOUT_MS, signal?: AbortSignal): Promise<T> {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let cancel = () => {};
  try {
    const cancelled = new Promise<never>((_resolve, reject) => {
      cancel = () => {
        controller.abort();
        reject(new DOMException("Read cancelled", "AbortError"));
      };
      signal?.addEventListener("abort", cancel, { once: true });
    });
    if (signal?.aborted) { cancel(); return await cancelled; }
    const deadline = new Promise<never>((_resolve, reject) => {
      timer = globalThis.setTimeout(() => {
        controller.abort();
        reject(new Error("Request timeout"));
      }, milliseconds);
    });
    return await Promise.race([operation(controller.signal), deadline, cancelled]);
  } finally {
    globalThis.clearTimeout(timer);
    signal?.removeEventListener("abort", cancel);
  }
}

// Limit dashboard bursts across all polling consumers. Queue wait is not a
// network failure and does not spend the read deadline. FIFO prevents starvation.
export const MAX_CONCURRENT_READS = 4;
let running = 0;
const queue: Array<() => void> = [];
function readSlot(signal: AbortSignal): Promise<() => void> {
  return new Promise((resolve, reject) => {
    const cancel = () => {
      const index = queue.indexOf(dispatch);
      if (index >= 0) queue.splice(index, 1);
      reject(new DOMException("Read cancelled", "AbortError"));
    };
    const dispatch = () => {
      signal.removeEventListener("abort", cancel);
      if (signal.aborted) { cancel(); return; }
      running++;
      let released = false;
      resolve(() => {
        if (released) return;
        released = true;
        running--;
        queue.shift()?.();
      });
    };
    if (signal.aborted) { cancel(); return; }
    signal.addEventListener("abort", cancel, { once: true });
    if (running < MAX_CONCURRENT_READS) dispatch();
    else queue.push(dispatch);
  });
}

// Wait a full interval after settlement, never overlap an in-flight read, and
// abort on disposal. Repeated manual refreshes while busy are coalesced.
export function pollRead<T>({ read, success, failure, start, interval = REFRESH_INTERVAL_MS, timeout = READ_TIMEOUT_MS }: {
  read: (signal: AbortSignal) => Promise<T>;
  success: (value: T) => void;
  failure: (error: unknown) => void;
  start?: () => void;
  interval?: number | null;
  timeout?: number;
}) {
  let active = true, inFlight = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const lifetime = new AbortController();
  const load = async () => {
    if (!active || inFlight) return;
    globalThis.clearTimeout(timer);
    inFlight = true;
    start?.();
    let release: (() => void) | undefined;
    try {
      release = await readSlot(lifetime.signal);
      const value = await deadlineRead(read, timeout, lifetime.signal);
      if (active) success(value);
    } catch (error) {
      if (active) failure(error);
    } finally {
      release?.();
      inFlight = false;
      if (active && interval !== null) timer = globalThis.setTimeout(() => void load(), interval);
    }
  };
  void load();
  return {
    refresh: () => void load(),
    stop: () => {
      active = false;
      globalThis.clearTimeout(timer);
      lifetime.abort();
    },
  };
}

export async function readJson(path: string, signal: AbortSignal, transport: typeof fetch = fetch): Promise<unknown> {
  const response = await transport(path, { cache: "no-store", headers: { Accept: "application/json" }, signal });
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return await response.json();
}
