"use client"

import Image from "next/image"
import FinGPT from "../assets/FinGPT2.png"
import { useChat } from "ai/react"
import { Message } from "ai"
import Bubble from "./components/bubble"
import Loading from "./components/loading"
import PromptSuggestion from "./components/promptSuggestion"


//download ai

const Home = () =>{

    const {append,isLoading,messages,input,handleInputChange,handleSubmit} = useChat()


    const noMessages = !messages || messages.length === 0

    const handlePrompt = (promptText)=>{
        const msg: Message = {
            id: crypto.randomUUID(),
            content:promptText,
            role: "user"
        }
        append(msg)
    }
    return (
        <main>
            <Image src={FinGPT} width = "250" alt ="FinGPT"/>
            <section className={noMessages ? "":"populated"}>
                {noMessages ?(
                    <>
                    <p className="starter-text">
                        This is Place for Financial terms!!
                        Ask FinGPT anything about Fintech and it will answer
                        back with most up-to-date answers.
                        I hope you njoy
                    </p>
                    <br/>
                    <PromptSuggestion onPromptClick={handlePrompt}/>
                    </>
                ) : (
                    <>
                        {messages.map((message,index)=> (
                            <Bubble key={`message-${index}`} message={message}/>
                        ))}
                        { isLoading &&<Loading/> }
                    </>
                )}
                <form onSubmit={handleSubmit}>
                    <input className="questionbox" onChange={handleInputChange} value={input} placeholder="Ask anything better..."/> 
                    <input type="submit"/>
                </form>
            </section>
        </main>
    ) 
}


export default Home