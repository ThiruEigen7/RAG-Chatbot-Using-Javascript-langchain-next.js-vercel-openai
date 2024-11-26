import "./global.css"

export const metadata = {
    title : "FinGPT",
    description : "The place to go for all your FinTech Questions"
}

const RootLayout = ({children}) =>{
    return (
        <html lang = "en">
            <body>{children}</body>
        </html>
    )
}

export default RootLayout