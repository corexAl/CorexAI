package main

import (
	"encoding/json"
	"net/http"
)

type Request struct {
	Prompt string `json:"prompt"`
}

type Response struct {
	Response string `json:"response"`
}

func chat(w http.ResponseWriter, r *http.Request) {

	var req Request

	json.NewDecoder(r.Body).Decode(&req)

	// Later this calls the Python COREX engine
	result := "COREX received: " + req.Prompt

	json.NewEncoder(w).Encode(Response{
		Response: result,
	})
}

func main() {

	http.HandleFunc("/api/chat", chat)

	println("COREX API running on :8080")

	http.ListenAndServe(":8080", nil)
}
