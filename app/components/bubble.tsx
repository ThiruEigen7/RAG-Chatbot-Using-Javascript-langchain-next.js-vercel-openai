const Bubble = ({message})=>{
    const {content,role} = message
    return (
        <div className={`${role} bub`}>{content}
        </div>
    )
}


export default Bubble