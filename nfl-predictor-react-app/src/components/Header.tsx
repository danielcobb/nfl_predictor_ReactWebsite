interface Props {
  children: string;
}
const Header = ({ children }: Props) => {
  return <h1 className="header">{children}</h1>;
};

export default Header;
