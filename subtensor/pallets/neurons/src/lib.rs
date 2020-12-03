#![cfg_attr(not(feature = "std"), no_std)]

// Frame imports.
use frame_support::{decl_module, decl_storage, decl_event, decl_error, dispatch, ensure, debug};
use frame_support::weights::{DispatchClass, Pays};
use codec::{Decode, Encode};
use frame_system::{ensure_signed};
use sp_std::{
	prelude::*
};

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

/// Configure the pallet by specifying the parameters and types on which it depends.
pub trait Trait: frame_system::Trait {
	/// Because this pallet emits events, it depends on the runtime's definition of an event.
	type Event: From<Event<Self>> + Into<<Self as frame_system::Trait>::Event>;
}

#[derive(Encode, Decode, Default)]
pub struct NeuronMetadata {
	ip: u128,
	port: u16,
	ip_type: u8,
}

// The pallet's runtime storage items.
decl_storage! {
	trait Store for Module<T: Trait> as SubtensorModule {
		// Active Neuron set: Active neurons in graph.
		pub Neurons get(fn neuron): map hasher(blake2_128_concat) T::AccountId => NeuronMetadata;

		// Active Neuron count.
		NeuronCount: u32;

	}
}

// Subtensor events.
decl_event!(
	pub enum Event<T> where AccountId = <T as frame_system::Trait>::AccountId {
		// Sent when a Neuron is added to the active set.
		NeuronAdded(AccountId),

		// Sent when a Neuron is removed from the active set.
		NeuronRemoved(AccountId),
	}
);

// Subtensor Errors.
decl_error! {
	pub enum Error for Module<T: Trait> {
		/// Cannot join as a member because you are already a member
		AlreadyActive,
		/// Cannot perform staking or emission unless the Neuron is already subscribed.
		NotActive,
		// Neuron calling emit has no emission.
		NothingToEmit,
		// Neuron updating weights caused overflow.
		WeightVecNotEqualSize,
		// Neuron setting weights are too large. Cause u32 overlfow.
		WeightSumToLarge,
	}
}

// Subtensor Dispatchable functions.
decl_module! {
	pub struct Module<T: Trait> for enum Call where origin: T::Origin {
		// Errors must be initialized if they are used by the pallet.
		type Error = Error<T>;

		// Events must be initialized if they are used by the pallet.
		fn deposit_event() = default;

		/// Emission. Called by an active Neuron with stake in-order to distribute 
		/// tokens to weighted neurons and to himself. The amount emitted is dependent on
		/// the ammount of stake held at this Neuron and the time since last emission.
		/// neurons are encouraged to calle this function often as to maximize
		/// their inflation in the graph.
		#[weight = (0, DispatchClass::Operational, Pays::No)]
		pub fn emit(origin) -> dispatch::DispatchResult {
			// Check that the extrinsic was signed and get the signer.
			// This function will return an error if the extrinsic is not signed.
			// https://substrate.dev/docs/en/knowledgebase/runtime/origin
			let calling_neuron = ensure_signed(origin)?;
			debug::info!("Emit sent by: {:?}", calling_neuron);

			// Check that the Neuron exists in the Neuron set.
			ensure!(Neurons::<T>::contains_key(&calling_neuron), Error::<T>::NotActive);

			// Return.
			Ok(())
		}

		// Subscribes the calling Neuron to the active set.
		#[weight = (0, DispatchClass::Operational, Pays::No)]
		fn subscribe(origin, ip: u128, port: u16, ip_type: u8) -> dispatch::DispatchResult {
			
			// Check sig.
			let new_neuron = ensure_signed(origin)?;
			debug::info!("new_neuron sent by: {:?}", new_neuron);

			// Check Neuron does not already exist.
			ensure!(!Neurons::<T>::contains_key(&new_neuron), Error::<T>::AlreadyActive);

			// Insert the new Neuron into the active set.
			Neurons::<T>::insert(&new_neuron, 
				NeuronMetadata {
					ip: ip,
					port: port,
					ip_type: ip_type,
				}
			);

			// Update Neuron count.
			let neuron_count = NeuronCount::get();
			NeuronCount::put(neuron_count + 1); // overflow check not necessary because of maximum
			debug::info!("neuron_count: {:?}", neuron_count + 1);

			// Emit event.
			Self::deposit_event(RawEvent::NeuronAdded(new_neuron));

			Ok(())
		}
		
		// Removes Neuron from active set.
		#[weight = (0, DispatchClass::Operational, Pays::No)]
		fn unsubscribe(origin) -> dispatch::DispatchResult {

			// Check sig.
			let old_neuron = ensure_signed(origin)?;
			debug::info!("unsubscribe sent by: {:?}", old_neuron);

			// Check that the Neuron already exists.
			ensure!(Neurons::<T>::contains_key(&old_neuron), Error::<T>::NotActive);
		
			// Remove Neuron.
			Neurons::<T>::remove(&old_neuron);
			NeuronCount::mutate(|v| *v -= 1);
			debug::info!("remove from Neuron set and decrement count.");

			Ok(())
		}
	}
}