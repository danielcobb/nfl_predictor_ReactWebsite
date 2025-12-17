import Header from "./components/Header";
import SelectMenu from "./components/SelectMenu";
import GameList from "./components/GameList";
import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  return (
    <div>
      <Header>NFL Game Predictor</Header>
      <SelectMenu />
      <GameList />
    </div>
  );
}

export default App;
