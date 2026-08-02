import { useState } from "react";
import { askCorex } from "./corex";

function App() {
    const [input, setInput] = useState("");
    const [output, setOutput] = useState("");

    async function send() {
        const result = await askCorex(input);
        setOutput(result);
    }

    return (
        <div>
            <h1>COREX</h1>

            <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
            />

            <button onClick={send}>
                Ask
            </button>

            <p>{output}</p>
        </div>
    );
}

export default App;
