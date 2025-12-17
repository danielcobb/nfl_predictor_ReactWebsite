interface Props {
  children: string;
}
const Header = ({ children }: Props) => {
  return (
    <>
      <header>
        <h1 className="title"> {children} </h1>
      </header>
    </>
  );
};

export default Header;
