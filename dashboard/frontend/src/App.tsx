import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/layout/Layout";
import Overview      from "./pages/Overview";
import Forecast      from "./pages/Forecast";
import CurrentWeek   from "./pages/CurrentWeek";
import EnsembleRouter from "./pages/EnsembleRouter";
import Tomorrow      from "./pages/Tomorrow";

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/"         element={<Overview />}       />
          <Route path="/forecast" element={<Forecast />}       />
          <Route path="/week"     element={<CurrentWeek />}    />
          <Route path="/shadow"   element={<EnsembleRouter />} />
          <Route path="/ensemble" element={<EnsembleRouter />} />
          <Route path="/tomorrow" element={<Tomorrow />}       />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
