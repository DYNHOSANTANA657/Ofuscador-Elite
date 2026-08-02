import React from "react";
import { createRoot } from "react-dom/client";
import { EliteApp } from "@/ui/EliteApp";
import "@/app/globals.css";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <EliteApp />
  </React.StrictMode>,
);
